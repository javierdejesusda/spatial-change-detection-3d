"""The fixed style sheet and viewer script of the report page.

Both are plain strings inlined into the page, so the page needs no network
and no other file. They avoid any URL, comment with two slashes, and call
that could reach the network.
"""

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f3f4f7;
  --surface: #ffffff;
  --sunken: #eceef2;
  --ink: #12161c;
  --muted: #4a5565;
  --line: rgba(18, 22, 28, 0.12);
  --shadow: 0 0 0 1px var(--line), 0 1px 2px rgba(18, 22, 28, 0.05),
    0 6px 16px rgba(18, 22, 28, 0.06);
  --image-outline: rgba(0, 0, 0, 0.1);
  --focus: #0b5cad;
  --canvas-bg: #f9fafc;
  --canvas-ink: #12161c;
  --canvas-grid: rgba(18, 22, 28, 0.16);
  --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1116;
    --surface: #171b22;
    --sunken: #1f242d;
    --ink: #e8ecf2;
    --muted: #a3adbb;
    --line: rgba(232, 236, 242, 0.14);
    --shadow: 0 0 0 1px var(--line), 0 1px 2px rgba(0, 0, 0, 0.3),
      0 6px 16px rgba(0, 0, 0, 0.3);
    --image-outline: rgba(255, 255, 255, 0.1);
    --focus: #74b3ff;
    --canvas-bg: #11151b;
    --canvas-ink: #e8ecf2;
    --canvas-grid: rgba(232, 236, 242, 0.2);
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 1rem/1.55 var(--font);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}
.skip {
  position: absolute;
  left: 16px;
  top: -64px;
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow);
}
.skip:focus-visible { top: 12px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.page { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
h1, h2, h3 { margin: 0; line-height: 1.2; text-wrap: balance; }
h1 { font-size: clamp(1.6rem, 4.5vw, 2.4rem); font-weight: 680;
  letter-spacing: -0.015em; }
h2 { font-size: 1.25rem; font-weight: 650; letter-spacing: -0.005em; }
h3 { font-size: 1.125rem; font-weight: 650; }
p { margin: 0; text-wrap: pretty; }
.lede { margin-top: 10px; max-width: 62ch; color: var(--muted); }
.num, dd, .tally-item .n { font-variant-numeric: tabular-nums; }
header.top { display: grid; gap: 20px; }
.tally-bar {
  display: flex;
  gap: 2px;
  height: 10px;
  border-radius: 5px;
  overflow: hidden;
  background: var(--sunken);
}
.tally-seg { flex: var(--n) 1 0; background: var(--c); min-width: 6px; }
.tally-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 22px;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}
.tally-item { display: inline-flex; align-items: center; gap: 8px; }
.tally-item .n { font-weight: 650; }
.tally-total { margin-top: 10px; color: var(--muted); }
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--c);
  flex: none;
}
section { margin-top: 48px; }
section > h2 { margin-bottom: 14px; }
.viewer {
  border-radius: 16px;
  background: var(--surface);
  box-shadow: var(--shadow);
  padding: 8px;
}
canvas {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 16 / 9;
  min-height: 280px;
  border-radius: 8px;
  background: var(--canvas-bg);
  touch-action: none;
  cursor: grab;
}
canvas:active { cursor: grabbing; }
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 16px;
  padding: 8px 4px 2px;
}
.toolbar label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  cursor: pointer;
}
.toolbar input { width: 20px; height: 20px; accent-color: var(--focus); }
.toolbar .spacer { flex: 1; }
.btn {
  min-width: 44px;
  min-height: 44px;
  padding: 0 16px;
  border: 0;
  border-radius: 10px;
  background: var(--sunken);
  color: var(--ink);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 0 0 1px var(--line);
  transition-property: transform, background-color, box-shadow;
  transition-duration: 150ms;
  transition-timing-function: cubic-bezier(0.2, 0, 0, 1);
}
.btn:active { transform: scale(0.96); }
@media (hover: hover) {
  .btn:hover { background: var(--line); }
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 18px;
  margin: 0;
  padding: 6px 4px 4px;
  list-style: none;
  color: var(--muted);
  font-size: 0.9rem;
}
.legend li { display: inline-flex; align-items: center; gap: 8px; }
.cards { display: grid; gap: 16px; padding: 0; margin: 0; }
.card {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);
  gap: 16px 24px;
  padding: 16px 16px 16px 24px;
  border-radius: 16px;
  background: var(--surface);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.card::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 6px;
  background: var(--c);
}
.card-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px 4px 10px;
  border-radius: 999px;
  background: var(--sunken);
  font-weight: 650;
  font-size: 0.9rem;
}
.cid { color: var(--muted); font-size: 0.9rem; }
.card h3 { margin-top: 10px; }
.card .desc { margin-top: 8px; max-width: 60ch; }
.facts, .evidence {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(78px, 1fr));
  gap: 10px 16px;
  margin: 14px 0 0;
}
.facts div, .evidence div { min-width: 0; }
dt { color: var(--muted); font-size: 0.85rem; }
dd { margin: 0; font-weight: 600; }
.evidence-title {
  margin-top: 16px;
  color: var(--muted);
  font-size: 0.85rem;
}
.evidence {
  margin-top: 6px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}
.crops { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 0; }
figure { margin: 0; min-width: 0; }
figcaption { margin-top: 6px; color: var(--muted); font-size: 0.85rem; }
.crop-frame {
  display: grid;
  place-items: center;
  aspect-ratio: 4 / 3;
  border-radius: 4px;
  background: var(--sunken);
  overflow: hidden;
}
.crop-frame img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  outline: 1px solid var(--image-outline);
  outline-offset: -1px;
}
.crop-frame.empty {
  padding: 12px;
  color: var(--muted);
  text-align: center;
  font-size: 0.9rem;
  box-shadow: inset 0 0 0 1px var(--line);
}
.card-actions { margin-top: 14px; }
.note, .empty-state {
  padding: 16px;
  border-radius: 12px;
  background: var(--surface);
  box-shadow: var(--shadow);
  color: var(--muted);
}
.regions { display: grid; gap: 8px; padding: 0; margin: 0; list-style: none; }
.regions li {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  padding: 12px 16px;
  border-radius: 12px;
  background: var(--surface);
  box-shadow: var(--shadow);
}
.regions .rid { font-weight: 650; }
.regions .rtext { color: var(--muted); }
.run { margin-top: 48px; color: var(--muted); font-size: 0.9rem; }
.run dl { display: flex; flex-wrap: wrap; gap: 4px 24px; margin: 8px 0 0; }
.run dl div { display: flex; gap: 8px; }
.run dd { font-weight: 500; }
@media (max-width: 640px) {
  .page { padding-top: 24px; }
  .card { grid-template-columns: minmax(0, 1fr); padding-left: 20px; }
  canvas { aspect-ratio: 4 / 3; }
}
@media (prefers-reduced-motion: reduce) {
  .btn { transition-duration: 0.01ms; }
}
"""

SCRIPT = """
(function () {
  var data = JSON.parse(document.getElementById('scene-data').textContent);
  var canvas = document.getElementById('view');
  var ctx = canvas.getContext('2d');
  if (!ctx) { return; }
  var center = data.view.center;
  var radius = data.view.radius;
  var home = { yaw: -0.8, pitch: 0.55, dist: radius * 2.4 };
  var cam = { yaw: home.yaw, pitch: home.pitch, dist: home.dist };
  var show = { before: true, after: true };
  var focusId = null;
  var pending = false;
  var theme = {};

  function readTheme() {
    var style = getComputedStyle(canvas);
    theme.ink = style.getPropertyValue('--canvas-ink').trim();
    theme.grid = style.getPropertyValue('--canvas-grid').trim();
  }

  var epochs = {};
  ['before', 'after'].forEach(function (name) {
    var e = data.epochs[name];
    var abgr = new Uint32Array(e.colors.length);
    for (var i = 0; i < abgr.length; i++) {
      var c = e.colors[i];
      abgr[i] = (255 << 24 | (c & 255) << 16 | (c >> 8 & 255) << 8 |
        (c >> 16 & 255)) >>> 0;
    }
    epochs[name] = { pts: Int32Array.from(e.points), abgr: abgr };
  });

  var near = radius * 0.05;

  function basis() {
    var cp = Math.cos(cam.pitch);
    var eye = [
      center[0] + cam.dist * cp * Math.cos(cam.yaw),
      center[1] + cam.dist * cp * Math.sin(cam.yaw),
      center[2] + cam.dist * Math.sin(cam.pitch)
    ];
    var f = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
    var n = Math.hypot(f[0], f[1], f[2]);
    f = [f[0] / n, f[1] / n, f[2] / n];
    var r = [f[1], -f[0], 0];
    var rn = Math.hypot(r[0], r[1]) || 1;
    r = [r[0] / rn, r[1] / rn, 0];
    var u = [
      r[1] * f[2] - r[2] * f[1],
      r[2] * f[0] - r[0] * f[2],
      r[0] * f[1] - r[1] * f[0]
    ];
    return { eye: eye, f: f, r: r, u: u };
  }

  function camera(b, x, y, z) {
    var dx = x - b.eye[0], dy = y - b.eye[1], dz = z - b.eye[2];
    return [
      dx * b.r[0] + dy * b.r[1] + dz * b.r[2],
      dx * b.u[0] + dy * b.u[1] + dz * b.u[2],
      dx * b.f[0] + dy * b.f[1] + dz * b.f[2]
    ];
  }

  function clip(a, c) {
    if (a[2] < near && c[2] < near) { return null; }
    if (a[2] < near || c[2] < near) {
      var t = (near - a[2]) / (c[2] - a[2]);
      var m = [a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1]), near];
      if (a[2] < near) { a = m; } else { c = m; }
    }
    return [a, c];
  }

  function screen(p, w, h, focal) {
    return [w / 2 + focal * p[0] / p[2], h / 2 - focal * p[1] / p[2]];
  }

  function drawSegments(b, segs, w, h, focal) {
    ctx.beginPath();
    segs.forEach(function (s) {
      var q = clip(camera(b, s[0], s[1], s[2]), camera(b, s[3], s[4], s[5]));
      if (!q) { return; }
      var a = screen(q[0], w, h, focal);
      var c = screen(q[1], w, h, focal);
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(c[0], c[1]);
    });
    ctx.stroke();
  }

  var image = null;
  var pixels = null;

  function frame(pw, ph) {
    if (!image || image.width !== pw || image.height !== ph) {
      image = ctx.createImageData(pw, ph);
      pixels = new Uint32Array(image.data.buffer);
    } else {
      pixels.fill(0);
    }
  }

  function drawPoints(b, w, h, focal, ratio) {
    var pw = canvas.width, ph = canvas.height;
    frame(pw, ph);
    var size = Math.max(1, Math.round(Math.min(3, w / 400) * ratio));
    var ex = b.eye[0], ey = b.eye[1], ez = b.eye[2];
    var f0 = b.f[0], f1 = b.f[1], f2 = b.f[2];
    var r0 = b.r[0], r1 = b.r[1], r2 = b.r[2];
    var u0 = b.u[0], u1 = b.u[1], u2 = b.u[2];
    ['before', 'after'].forEach(function (name) {
      if (!show[name]) { return; }
      var pts = epochs[name].pts, abgr = epochs[name].abgr;
      for (var i = 0; i < abgr.length; i++) {
        var dx = pts[3 * i] - ex, dy = pts[3 * i + 1] - ey;
        var dz = pts[3 * i + 2] - ez;
        var depth = dx * f0 + dy * f1 + dz * f2;
        if (depth < near) { continue; }
        var x = Math.round((w / 2 + focal * (dx * r0 + dy * r1 + dz * r2) /
          depth) * ratio);
        var y = Math.round((h / 2 - focal * (dx * u0 + dy * u1 + dz * u2) /
          depth) * ratio);
        for (var j = 0; j < size; j++) {
          for (var k = 0; k < size; k++) {
            if (x + k >= 0 && x + k < pw && y + j >= 0 && y + j < ph) {
              pixels[(y + j) * pw + x + k] = abgr[i];
            }
          }
        }
      }
    });
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.putImageData(image, 0, 0);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function draw() {
    pending = false;
    var ratio = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    var pw = Math.round(w * ratio), ph = Math.round(h * ratio);
    if (pw === 0 || ph === 0) { return; }
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    var b = basis();
    var focal = h * 1.1;
    drawPoints(b, w, h, focal, ratio);
    ctx.lineJoin = 'round';
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    data.regions.forEach(function (r) {
      drawSegments(b, r.segments, w, h, focal);
    });
    data.changes.forEach(function (c) {
      var dim = focusId !== null && focusId !== c.id;
      ctx.globalAlpha = dim ? 0.25 : 1;
      ctx.strokeStyle = data.colors[c.type];
      c.lines.forEach(function (line) {
        ctx.lineWidth = line.role === 'arrow' || focusId === c.id ? 3 : 2;
        ctx.setLineDash(line.role === 'before_box' ? [7, 5] : []);
        drawSegments(b, line.segments, w, h, focal);
      });
    });
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    if (!epochs.before.abgr.length && !epochs.after.abgr.length &&
        !data.changes.length) {
      ctx.fillStyle = theme.ink;
      ctx.textAlign = 'center';
      ctx.font = '15px ' + getComputedStyle(canvas).fontFamily;
      ctx.fillText('Nothing to show in 3D', w / 2, h / 2);
    }
  }

  function redraw() {
    if (!pending) { pending = true; requestAnimationFrame(draw); }
  }

  function limit() {
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch));
    cam.dist = Math.max(radius * 0.2, Math.min(radius * 8, cam.dist));
    redraw();
  }

  function reset() {
    cam.yaw = home.yaw;
    cam.pitch = home.pitch;
    cam.dist = home.dist;
    focusId = null;
    limit();
  }

  var drag = null;
  canvas.addEventListener('pointerdown', function (ev) {
    drag = { x: ev.clientX, y: ev.clientY };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', function (ev) {
    if (!drag) { return; }
    cam.yaw -= (ev.clientX - drag.x) * 0.008;
    cam.pitch += (ev.clientY - drag.y) * 0.008;
    drag = { x: ev.clientX, y: ev.clientY };
    limit();
  });
  function endDrag() { drag = null; }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', function (ev) {
    ev.preventDefault();
    cam.dist *= Math.exp(ev.deltaY * 0.001);
    limit();
  }, { passive: false });
  canvas.addEventListener('keydown', function (ev) {
    var k = ev.key;
    var handled = true;
    if (k === 'ArrowLeft') { cam.yaw += 0.1; }
    else if (k === 'ArrowRight') { cam.yaw -= 0.1; }
    else if (k === 'ArrowUp') { cam.pitch += 0.1; }
    else if (k === 'ArrowDown') { cam.pitch -= 0.1; }
    else if (k === '+' || k === '=') { cam.dist *= 0.85; }
    else if (k === '-') { cam.dist /= 0.85; }
    else if (k === '0' || k === 'Home') { reset(); }
    else { handled = false; }
    if (handled) { ev.preventDefault(); limit(); }
  });

  function bindToggle(id, name) {
    document.getElementById(id).addEventListener('change', function (ev) {
      show[name] = ev.target.checked;
      redraw();
    });
  }
  bindToggle('show-before', 'before');
  bindToggle('show-after', 'after');
  document.getElementById('zoom-in').addEventListener('click', function () {
    cam.dist *= 0.8; limit();
  });
  document.getElementById('zoom-out').addEventListener('click', function () {
    cam.dist /= 0.8; limit();
  });
  document.getElementById('reset').addEventListener('click', reset);
  document.querySelectorAll('[data-focus]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      focusId = btn.getAttribute('data-focus');
      redraw();
      canvas.scrollIntoView({ block: 'nearest' });
    });
  });

  readTheme();
  var scheme = window.matchMedia('(prefers-color-scheme: dark)');
  scheme.addEventListener('change', function () {
    readTheme();
    redraw();
  });
  if (window.ResizeObserver) { new ResizeObserver(redraw).observe(canvas); }
  window.addEventListener('resize', redraw);
  redraw();
})();
"""
