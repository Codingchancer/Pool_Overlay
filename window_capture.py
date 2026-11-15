"""Capture one window's own picture (the mirror), without the windows on top of it.

The guide overlay sits on top of the AirPlay window. A screen capture sees the
guide too, which then has to be un-blended from every frame (``remove_overlay``
/ ``shown_layer``) and ties each capture to what was last drawn. Copying the
mirror window's own picture leaves the guide (and the control panel) out
entirely, and it is much faster: ~1 ms instead of ~8 ms for a screen grab.

Two Win32 ways are tried, fastest first, and the first that gives a real
picture is kept:

* ``blit``: BitBlt from the window's own device context. Under the desktop
  compositor this is the window's own surface, so other windows never appear.
* ``print``: PrintWindow with PW_RENDERFULLCONTENT, which also works for
  windows drawn with DirectX (video players) where ``blit`` gives black.

If neither gives a picture, ``grab`` returns None and the caller falls back to
screen capture. The GDI objects are created once and reused for every frame,
and released by ``close`` (one capture per worker thread; see issue #13 for
what leaking them does).
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

if sys.platform == "win32":
    from ctypes import wintypes

    # Private library objects: ctypes.windll.user32 is shared by the whole
    # app, and other code (overlay.py) sets its own argtypes on the same
    # functions, which made these calls fail with "wrong type" errors.
    _user32 = ctypes.WinDLL("user32")
    _gdi32 = ctypes.WinDLL("gdi32")
    _dwmapi = ctypes.WinDLL("dwmapi")
    _user32.GetDC.restype = wintypes.HDC
    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.GetWindow.restype = wintypes.HWND
    _user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    _user32.GetTopWindow.restype = wintypes.HWND
    _user32.GetTopWindow.argtypes = [wintypes.HWND]
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _gdi32.CreateCompatibleDC.restype = wintypes.HDC
    _gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    _gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    _gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
                                        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
    _gdi32.SelectObject.restype = wintypes.HGDIOBJ
    _gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    _gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    _gdi32.DeleteDC.argtypes = [wintypes.HDC]
    _gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]

    class _BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

_SRCCOPY = 0x00CC0020
_PW_CLIENTONLY = 1
_PW_RENDERFULLCONTENT = 2
_GW_HWNDNEXT = 2
_DWMWA_CLOAKED = 14
# Outside the window (if the guide area reaches past it): neutral grey, which
# the detector treats as neither cloth, pocket nor white guide.
_OUTSIDE = 128


def _client_rect(hwnd) -> tuple[int, int, int, int] | None:
    """The window's drawing area in physical screen pixels: (left, top, width, height)."""
    rect = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    return origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top


def _cloaked(hwnd) -> bool:
    value = ctypes.c_int(0)
    try:
        _dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), _DWMWA_CLOAKED, ctypes.byref(value),
                                      ctypes.sizeof(value))
    except Exception:
        return False
    return bool(value.value)


# Mirror windows known by title: chosen even when another window covers them
# (that is the point of capturing the window itself).
MIRROR_TITLES = ("AirPlay Video Stream", "uxplay-windows")
MIRROR_TITLE_PREFIXES = ("Synthetic",)  # the test tables (fake_game.py and tools)


def _title(hwnd) -> str:
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def window_under(rect: dict, skip_process: int | None = None) -> int | None:
    """The mirror window under the middle of ``rect`` (physical px).

    Windows of ``skip_process`` (this app: guide, control panel, simulator)
    are passed over. Of the windows covering the middle, a known mirror
    window (by title) wins; otherwise the top-most one.
    """
    if sys.platform != "win32":
        return None
    skip_process = os.getpid() if skip_process is None else skip_process
    cx = rect["left"] + rect["width"] // 2
    cy = rect["top"] + rect["height"] // 2
    first = None
    hwnd = _user32.GetTopWindow(None)
    while hwnd:
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != skip_process and _user32.IsWindowVisible(hwnd) and not _user32.IsIconic(hwnd) \
                and not _cloaked(hwnd):
            client = _client_rect(hwnd)
            if client and client[2] > 0 and client[3] > 0 and \
                    client[0] <= cx < client[0] + client[2] and client[1] <= cy < client[1] + client[3]:
                title = _title(hwnd)
                if title in MIRROR_TITLES or title.startswith(MIRROR_TITLE_PREFIXES):
                    return int(hwnd)
                if first is None:
                    first = int(hwnd)
        hwnd = _user32.GetWindow(hwnd, _GW_HWNDNEXT)
    return first


class WindowCapture:
    """Copies ``hwnd``'s own picture. Open once, ``grab`` every frame, ``close`` at the end."""

    # After no method gave a picture, wait this long before trying again (the
    # caller uses screen capture meanwhile; trying both costs ~15 ms a frame).
    RETRY_SECONDS = 2.0

    def __init__(self, hwnd: int):
        self.hwnd = int(hwnd)
        self.method: str | None = None      # "blit" / "print" once one has worked
        self._window_dc = None
        self._dc = None
        self._bitmap = None
        self._old = None
        self._view = None
        self._size = (0, 0)
        self.closed = False
        self._failed_at = None

    def _ensure_buffer(self, width: int, height: int) -> bool:
        if self._view is not None and self._size == (width, height):
            return True
        self._release_buffer()
        if self._window_dc is None:
            self._window_dc = _user32.GetDC(self.hwnd)
            if not self._window_dc:
                self._window_dc = None
                return False
        self._dc = _gdi32.CreateCompatibleDC(self._window_dc)
        header = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        bits = ctypes.c_void_p()
        self._bitmap = _gdi32.CreateDIBSection(self._dc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
        if not self._dc or not self._bitmap or not bits.value:
            self._release_buffer()
            return False
        self._old = _gdi32.SelectObject(self._dc, self._bitmap)
        self._view = np.ctypeslib.as_array(ctypes.cast(bits, ctypes.POINTER(ctypes.c_uint8)), (height, width, 4))
        self._size = (width, height)
        return True

    def _copy(self, method: str, width: int, height: int) -> bool:
        if method == "blit":
            return bool(_gdi32.BitBlt(self._dc, 0, 0, width, height, self._window_dc, 0, 0, _SRCCOPY))
        return bool(_user32.PrintWindow(self.hwnd, self._dc, _PW_CLIENTONLY | _PW_RENDERFULLCONTENT))

    @staticmethod
    def _has_picture(view: np.ndarray) -> bool:
        # A failed copy is all black (or all one colour); a pool table never is.
        sample = view[::16, ::16, :3]
        return int(sample.max()) > 24 and float(sample.std()) > 4

    def grab(self, region: dict) -> np.ndarray | None:
        """``region`` (physical screen px) as a BGR image from the window's own picture.

        Parts of the region outside the window are neutral grey. Returns None
        when the window is gone or no copy method gives a picture.
        """
        if self.closed or sys.platform != "win32" or not _user32.IsWindow(self.hwnd) \
                or _user32.IsIconic(self.hwnd):
            return None
        if self._failed_at is not None and time.monotonic() - self._failed_at < self.RETRY_SECONDS:
            return None
        client = _client_rect(self.hwnd)
        if client is None or client[2] <= 0 or client[3] <= 0:
            return None
        left, top, width, height = client
        if not self._ensure_buffer(width, height):
            return None
        # The method that worked last time first; the other only if it stops
        # giving a picture (e.g. the window switched to DirectX drawing).
        methods = ["blit", "print"]
        if self.method in methods:
            methods.remove(self.method)
            methods.insert(0, self.method)
        for method in methods:
            if self._copy(method, width, height) and self._has_picture(self._view):
                self.method = method
                break
        else:
            self.method = None
            self._failed_at = time.monotonic()
            return None
        self._failed_at = None
        out = np.full((region["height"], region["width"], 3), _OUTSIDE, np.uint8)
        x0, y0 = max(region["left"], left), max(region["top"], top)
        x1 = min(region["left"] + region["width"], left + width)
        y1 = min(region["top"] + region["height"], top + height)
        if x1 > x0 and y1 > y0:
            out[y0 - region["top"]:y1 - region["top"], x0 - region["left"]:x1 - region["left"]] = \
                self._view[y0 - top:y1 - top, x0 - left:x1 - left, :3]
        return out

    def _release_buffer(self):
        if self._dc:
            if self._old:
                _gdi32.SelectObject(self._dc, self._old)
            if self._bitmap:
                _gdi32.DeleteObject(self._bitmap)
            _gdi32.DeleteDC(self._dc)
        self._dc = self._bitmap = self._old = self._view = None
        self._size = (0, 0)

    def close(self):
        """Release every GDI object this capture holds."""
        self._release_buffer()
        if self._window_dc:
            _user32.ReleaseDC(self.hwnd, self._window_dc)
        self._window_dc = None
        self.closed = True
