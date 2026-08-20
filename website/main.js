/* UniVis gallery v2.1 — richer lattice pulses, self-resuming hero with
   per-layer normalization + "UniVis" write-in, hover previews, dual theme. */

(function () {
  'use strict';

  /* ================= theme ================= */
  var root = document.documentElement;
  var themeBtn = document.getElementById('themeToggle');
  function currentTheme() { return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark'; }
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('univis-theme', next); } catch (e) {}
      document.dispatchEvent(new CustomEvent('themechange'));
    });
  }
  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  /* ================= plasma ================= */
  var STOPS = [
    [0.00, 13, 8, 135],
    [0.25, 126, 3, 168],
    [0.50, 204, 71, 120],
    [0.75, 248, 148, 65],
    [1.00, 240, 249, 33],
  ];
  function plasmaRGB(t) {
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i];
        var f = (t - a[0]) / (b[0] - a[0]);
        return [0, 1, 2].map(function (k) {
          return Math.round(a[k + 1] + f * (b[k + 1] - a[k + 1]));
        });
      }
    }
    return [240, 249, 33];
  }
  function plasma(t) { return 'rgb(' + plasmaRGB(t).join(',') + ')'; }

  /* ================= wafer compute lattice background ================= */
  var bg = document.getElementById('bgCanvas');
  if (bg) {
    var bctx = bg.getContext('2d');
    var DPR = Math.min(window.devicePixelRatio || 1, 2);
    var CELL = 30;
    var staticLayer = null;
    var pulses = [];
    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function sizeBg() {
      bg.width = Math.floor(innerWidth * DPR);
      bg.height = Math.floor(innerHeight * DPR);
    }

    function mulberry32(seed) {
      return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }

    function renderStatic() {
      var W = bg.width, H = bg.height;
      var off = document.createElement('canvas');
      off.width = W; off.height = H;
      var c = off.getContext('2d');
      var rnd = mulberry32(20260819);
      var line = cssVar('--lat-line');
      var line2 = cssVar('--lat-line2');
      var trace = cssVar('--lat-trace');
      var via = cssVar('--lat-via');

      var cols = Math.ceil(W / (CELL * DPR)), rows = Math.ceil(H / (CELL * DPR));
      for (var i = 0; i <= cols; i++) {
        for (var j = 0; j <= rows; j++) {
          var x = i * CELL * DPR, y = j * CELL * DPR;
          c.strokeStyle = (rnd() < 0.06) ? line2 : line;
          c.lineWidth = 1;
          c.strokeRect(x + 4 * DPR, y + 4 * DPR, CELL * DPR - 8 * DPR, CELL * DPR - 8 * DPR);
        }
      }
      var nTraces = Math.max(6, Math.floor(cols * rows / 260));
      for (var k = 0; k < nTraces; k++) {
        var x0 = Math.floor(rnd() * cols) * CELL * DPR + CELL * DPR / 2;
        var y0 = Math.floor(rnd() * rows) * CELL * DPR + CELL * DPR / 2;
        var segs = 2 + Math.floor(rnd() * 3);
        var horiz = rnd() < 0.5;
        c.strokeStyle = trace; c.lineWidth = 1.2 * DPR;
        c.beginPath(); c.moveTo(x0, y0);
        var cx = x0, cy = y0;
        for (var sgi = 0; sgi < segs; sgi++) {
          var len = (1 + Math.floor(rnd() * 4)) * CELL * DPR * (rnd() < 0.5 ? -1 : 1);
          if (horiz) { cx += len; } else { cy += len; }
          c.lineTo(cx, cy);
          horiz = !horiz;
        }
        c.stroke();
        c.fillStyle = via;
        [[x0, y0], [cx, cy]].forEach(function (p) {
          c.beginPath(); c.arc(p[0], p[1], 2.2 * DPR, 0, Math.PI * 2); c.fill();
        });
      }
      c.strokeStyle = line2; c.lineWidth = 1.4 * DPR;
      var m = 26 * DPR, L = 60 * DPR;
      [[m, m, 1, 1], [W - m, m, -1, 1], [m, H - m, 1, -1], [W - m, H - m, -1, -1]].forEach(function (q) {
        c.beginPath();
        c.moveTo(q[0] + q[2] * L, q[1]); c.lineTo(q[0], q[1]); c.lineTo(q[0], q[1] + q[3] * L);
        c.stroke();
      });
      staticLayer = off;
    }

    function drawPulse(p, now) {
      var age = (now - p.t0) / p.dur;
      if (age < 0 || age > 1) return false;
      var ease = age < 0.5 ? age * 2 : (1 - age) * 2;
      var rgb = plasmaRGB(p.hue);
      /* each cell of a cluster gets a nearby hue: multi-color within one pulse */
      bctx.globalAlpha = 0.13 * ease;
      p.cells.forEach(function (cl, idx) {
        var h2 = Math.min(1, Math.max(0, p.hue + cl[2]));
        var c2 = plasmaRGB(h2);
        bctx.fillStyle = 'rgb(' + c2.join(',') + ')';
        bctx.fillRect(cl[0] * CELL * DPR + 4 * DPR, cl[1] * CELL * DPR + 4 * DPR,
          CELL * DPR - 8 * DPR, CELL * DPR - 8 * DPR);
      });
      bctx.globalAlpha = 1;
      return true;
    }

    function loop(now) {
      bctx.clearRect(0, 0, bg.width, bg.height);
      if (staticLayer) bctx.drawImage(staticLayer, 0, 0);
      pulses = pulses.filter(function (p) { return drawPulse(p, now); });
      if (pulses.length < 6 && Math.random() < 0.06) {
        var cols = Math.ceil(bg.width / (CELL * DPR)), rows = Math.ceil(bg.height / (CELL * DPR));
        var cx = 1 + Math.floor(Math.random() * (cols - 4));
        var cy = 1 + Math.floor(Math.random() * (rows - 4));
        var hue = Math.random();
        var cells = [];
        var n = 4 + Math.floor(Math.random() * 5);
        for (var i = 0; i < n; i++) {
          cells.push([
            cx + Math.floor(Math.random() * 3),
            cy + Math.floor(Math.random() * 3),
            (Math.random() - 0.5) * 0.3, /* per-cell hue offset */
          ]);
        }
        pulses.push({ cells: cells, hue: hue, t0: now, dur: 2000 + Math.random() * 1600 });
      }
      requestAnimationFrame(loop);
    }

    function initBg() {
      sizeBg();
      renderStatic();
      bctx.clearRect(0, 0, bg.width, bg.height);
      bctx.drawImage(staticLayer, 0, 0);
      if (!reduced && innerWidth >= 768 && !document.hidden) {
        requestAnimationFrame(loop);
      }
    }
    var rszTimer = null;
    window.addEventListener('resize', function () {
      clearTimeout(rszTimer);
      rszTimer = setTimeout(initBg, 200);
    });
    document.addEventListener('themechange', initBg);
    initBg();
  }

  /* ================= hero telemetry replay ================= */
  var canvas = document.getElementById('heroCanvas');
  if (canvas && typeof HERO !== 'undefined') {
    var ctx = canvas.getContext('2d');
    var ROWS = HERO.layers, COLS = Math.min(HERO.tokens, 478);
    var DATA = HERO.data;

    /* per-layer robust scaling: each layer normalized to its own p2-p98 so
       mid/low layers' structure is visible next to the dominant first layer */
    var rowLo = [], rowSpan = [];
    for (var l = 0; l < ROWS; l++) {
      var vals = [];
      for (var s = 0; s < COLS; s++) vals.push(DATA[s][l]);
      vals.sort(function (a, b) { return a - b; });
      var lo2 = vals[Math.floor(vals.length * 0.02)];
      var hi2 = vals[Math.floor(vals.length * 0.98)];
      rowLo.push(lo2);
      rowSpan.push(Math.max(hi2 - lo2, 1e-6));
    }
    function valAt(s, l) {
      return Math.max(0, Math.min(1, (DATA[s][l] - rowLo[l]) / rowSpan[l]));
    }

    var W = canvas.width, H = canvas.height;
    var padL = 46, padR = 10, padT = 12, padB = 26;
    var cw = (W - padL - padR) / COLS, ch = (H - padT - padB) / ROWS;

    /* "UniVis" cell mask: rasterize the word onto the cell grid, placed in
       the low-activity (dark/blue) region on the right half */
    var NAME = 'UniVis';
    var nameMask = []; /* [col, row] cells */
    (function buildNameMask() {
      var off = document.createElement('canvas');
      var gw = Math.floor((W - padL - padR) / cw * 0.92);
      var gh = ROWS;
      off.width = gw; off.height = gh;
      var c = off.getContext('2d');
      c.fillStyle = '#fff';
      var fs = Math.min(gh * 0.52, 34);
      c.font = '700 ' + fs + 'px "DM Sans", Arial, sans-serif';
      c.textBaseline = 'middle';
      c.textAlign = 'center';
      c.fillText(NAME, gw / 2, gh / 2 + 1);
      var img = c.getImageData(0, 0, gw, gh).data;
      var startCol = Math.floor(COLS * 0.58);
      var avail = Math.min(gw, COLS - startCol - 2);
      for (var y = 0; y < gh; y++) {
        for (var x = 0; x < avail; x++) {
          if (img[(y * gw + x) * 4 + 3] > 128) {
            nameMask.push([startCol + x, y]);
          }
        }
      }
    })();
    var nameCellsByCol = {};
    nameMask.forEach(function (m) {
      (nameCellsByCol[m[0]] = nameCellsByCol[m[0]] || []).push(m[1]);
    });
    var nameCols = Object.keys(nameCellsByCol).map(Number).sort(function (a, b) { return a - b; });

    function drawGrid() {
      ctx.fillStyle = '#0d1017';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.fillStyle = 'rgba(152,161,179,0.75)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.lineWidth = 1;
      for (var l = 0; l <= ROWS; l += 4) {
        var y = padT + l * ch;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillText(String(l), 16, y + 3);
      }
      ctx.fillText('layer', 6, padT - 3);
    }

    function cellRect(s, l) {
      return [padL + s * cw, padT + l * ch, Math.max(cw - 0.4, 0.6), Math.max(ch - 0.6, 0.6)];
    }

    /* mode: 'data' up to col; nameProg 0..1 writes UniVis cells hot */
    function drawUpTo(col, nameProg) {
      drawGrid();
      for (var s = 0; s < Math.min(col, COLS); s++) {
        for (var l = 0; l < ROWS; l++) {
          var r = cellRect(s, l);
          ctx.fillStyle = plasma(valAt(s, l));
          ctx.fillRect(r[0], r[1], r[2], r[3]);
        }
      }
      if (col > 0 && col < COLS) {
        ctx.fillStyle = 'rgba(240,249,33,0.16)';
        ctx.fillRect(padL + (col - 1) * cw, padT, cw * 2.2, ROWS * ch);
      }
      /* UniVis write-in: letters emerge as hot cells over the quiet region */
      if (nameProg > 0) {
        var upto = Math.floor(nameProg * nameCols.length);
        for (var i = 0; i < upto; i++) {
          var nc = nameCols[i];
          nameCellsByCol[nc].forEach(function (rw) {
            var r = cellRect(nc, rw);
            var t = 0.72 + 0.28 * ((nc % 7) / 7);
            ctx.fillStyle = plasma(t);
            ctx.fillRect(r[0], r[1], r[2], r[3]);
          });
        }
        /* leading glow edge */
        if (upto > 0 && upto < nameCols.length) {
          var ec = nameCols[upto - 1];
          ctx.fillStyle = 'rgba(240,249,33,0.25)';
          ctx.fillRect(padL + ec * cw, padT, cw * 2.5, ROWS * ch);
        }
      }
      var lbl = document.getElementById('roToken');
      var st = document.getElementById('roState');
      if (lbl) {
        lbl.textContent = nameProg > 0
          ? 'generating: ' + NAME
          : 'token ' + String(Math.min(col, COLS)).padStart(3, '0') + '/' + COLS;
      }
      if (st && nameProg >= 1) st.textContent = 'signature · ' + NAME;
    }

    var reducedHero = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedHero) {
      drawUpTo(COLS, 1);
      var st0 = document.getElementById('roState');
      if (st0) st0.textContent = 'static render';
    } else {
      var col = 0, nameProg = 0, holdUntil = 0, phase = 'data'; /* data -> name -> hold */
      var lastPointer = 0, paused = false;

      function frame(ts) {
        if (!paused && ts > holdUntil) {
          if (phase === 'data') {
            var speed = Math.max(1, Math.round(COLS / 900));
            drawUpTo(col, 0);
            col += speed;
            if (col > COLS + 30) { phase = 'name'; col = COLS; }
          } else if (phase === 'name') {
            nameProg = Math.min(1, nameProg + 0.02);
            drawUpTo(COLS, nameProg);
            if (nameProg >= 1) { phase = 'hold'; holdUntil = ts + 3200; }
          } else {
            drawUpTo(COLS, 1);
            holdUntil = ts + 500;
            phase = 'data'; col = 0; nameProg = 0;
          }
        }
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);

      function scrub(e) {
        paused = true;
        lastPointer = Date.now();
        var rect = canvas.getBoundingClientRect();
        var x = (e.clientX - rect.left) / rect.width * W;
        var c = Math.round((x - padL) / cw);
        phase = 'data'; nameProg = 0;
        drawUpTo(Math.max(0, Math.min(c, COLS)), 0);
        var st = document.getElementById('roState');
        if (st) st.textContent = 'scrub — pointer';
      }
      canvas.addEventListener('pointermove', scrub);
      canvas.addEventListener('pointerdown', scrub);
      /* self-resume: 2.5s after the last pointer interaction */
      setInterval(function () {
        if (paused && Date.now() - lastPointer > 2500) {
          paused = false;
          phase = 'data';
          var st = document.getElementById('roState');
          if (st) st.textContent = 'replaying…';
        }
      }, 500);
    }
  }

  /* ================= report cards + hover preview ================= */
  var REPORTS = [
    { f: 'c500_phase_a', hw: 'c500', m: 'Qwen2.5-0.5B', note: '阶段 A 通路验证 · 3 提示词 × 128 token · 24 层 × 384 步 · 零代码修改' },
    { f: 'c500_0.5B', hw: 'c500', m: 'Qwen2.5-0.5B', note: '阶段 B 基线协议 · 50 token · 与 L20 同题对照' },
    { f: 'l20_0.5B', hw: 'l20', m: 'Qwen2.5-0.5B', note: '阶段 B 基线协议 · 50 token · r = 1.0000' },
    { f: 'c500_3B', hw: 'c500', m: 'Qwen2.5-3B', note: '36 层 · 6.3GB bf16 · 与 L20 画像一致' },
    { f: 'l20_3B', hw: 'l20', m: 'Qwen2.5-3B', note: '36 层 · r = 1.0000 · MAE 0.0007' },
    { f: 'c500_7B', hw: 'c500', m: 'Qwen2.5-7B', note: '15.3GB bf16 跑通 16GB 切分实例 · 28 层' },
    { f: 'l20_7B', hw: 'l20', m: 'Qwen2.5-7B', note: 'r = 1.0000 · MAE 0.0005（三规模最紧）' },
  ];
  var grid = document.getElementById('reportGrid');
  var preview = document.getElementById('thumbPreview');
  if (grid) {
    REPORTS.forEach(function (r) {
      var a = document.createElement('a');
      a.className = 'rcard';
      a.href = 'assets/reports/' + r.f + '.html';
      a.target = '_blank';
      a.rel = 'noopener';
      a.dataset.thumb = 'assets/thumbs/' + r.f + '.png';
      a.dataset.model = r.m + ' · ' + (r.hw === 'c500' ? 'MetaX C500' : 'NVIDIA L20');
      a.innerHTML =
        '<span class="tag ' + r.hw + '">' + (r.hw === 'c500' ? 'MetaX C500 · MXMACA' : 'NVIDIA L20') + '</span>' +
        '<h3>' + r.m + '</h3>' +
        '<p>' + r.note + '</p>' +
        '<span class="spark" aria-hidden="true"><i style="width:' +
        (60 + Math.random() * 40).toFixed(0) + '%;background:linear-gradient(90deg,#7e03a8,#cc4778,#f89441)"></i></span>' +
        '<span class="open">打开完整报告 ↗</span>';
      grid.appendChild(a);

      if (preview && matchMedia('(hover: hover)').matches) {
        a.addEventListener('mouseenter', function () {
          var img = preview.querySelector('img');
          img.src = a.dataset.thumb;
          preview.querySelector('.tp-cap').textContent = a.dataset.model + ' · 点击卡片打开完整报告';
          var rect = a.getBoundingClientRect();
          var px = Math.min(Math.max(rect.left, 12), innerWidth - 352);
          var py = rect.bottom + 10;
          if (py + 280 > innerHeight) py = Math.max(12, rect.top - 288);
          preview.style.left = px + 'px';
          preview.style.top = py + 'px';
          preview.classList.add('show');
        });
        a.addEventListener('mouseleave', function () { preview.classList.remove('show'); });
      }
    });
  }

  /* ================= early-exit chart (theme-aware) ================= */
  var EE = {
    q05: { base: 101.3, th: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], tok: [99.4, 95.3, 78.9, 73.0, 47.5, 21.9], kept: [0.96, 0.93, 0.79, 0.74, 0.54, 0.30], color: '#f89441' },
    q3: { base: 102.0, th: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], tok: [79.8, 70.3, 64.5, 47.4, 24.5, 8.5], kept: [0.79, 0.72, 0.67, 0.53, 0.33, 0.17], color: '#cc4778' },
  };
  var eeCanvas = document.getElementById('eeChart');
  if (eeCanvas) {
    var ectx = eeCanvas.getContext('2d');
    var EW = eeCanvas.width, EH = eeCanvas.height;
    var epad = { l: 56, r: 20, t: 18, b: 44 };
    var model = 'q05';
    var hoverIdx = -1;

    function xOf(th) {
      var lo2 = Math.log(0.05), hi2 = Math.log(2.0);
      return epad.l + ((Math.log(th) - lo2) / (hi2 - lo2)) * (EW - epad.l - epad.r);
    }
    function yOf(pct) { return EH - epad.b - (pct / 100) * (EH - epad.t - epad.b); }

    function drawEE() {
      var d = EE[model];
      var chartBg = cssVar('--chart-bg') || '#0d1017';
      var gridC = cssVar('--chart-grid') || 'rgba(255,255,255,0.1)';
      var labelC = cssVar('--chart-label') || 'rgba(152,161,179,0.85)';
      ectx.fillStyle = chartBg;
      ectx.fillRect(0, 0, EW, EH);
      ectx.strokeStyle = gridC;
      ectx.fillStyle = labelC;
      ectx.font = '11px JetBrains Mono, monospace';
      for (var p = 0; p <= 100; p += 25) {
        var y = yOf(p);
        ectx.beginPath(); ectx.moveTo(epad.l, y); ectx.lineTo(EW - epad.r, y); ectx.stroke();
        ectx.fillText(p + '%', 14, y + 4);
      }
      d.th.forEach(function (th) {
        var x = xOf(th);
        ectx.beginPath(); ectx.moveTo(x, EH - epad.b); ectx.lineTo(x, EH - epad.b + 5); ectx.stroke();
        ectx.fillText(String(th), x - 10, EH - epad.b + 20);
      });
      ectx.fillText('熵阈值 (nats, log)', EW / 2 - 60, EH - 8);
      ectx.save();
      ectx.translate(14, EH / 2 + 40);
      ectx.rotate(-Math.PI / 2);
      ectx.fillText('生成 token / 基线', 0, 0);
      ectx.restore();
      ectx.strokeStyle = 'rgba(128,128,128,0.35)';
      ectx.setLineDash([4, 4]);
      ectx.beginPath(); ectx.moveTo(epad.l, yOf(100)); ectx.lineTo(EW - epad.r, yOf(100)); ectx.stroke();
      ectx.setLineDash([]);
      ectx.strokeStyle = d.color;
      ectx.lineWidth = 2;
      ectx.beginPath();
      d.th.forEach(function (th, i) {
        var x = xOf(th), y = yOf(d.tok[i] / d.base * 100);
        if (i === 0) ectx.moveTo(x, y); else ectx.lineTo(x, y);
      });
      ectx.stroke();
      d.th.forEach(function (th, i) {
        var x = xOf(th), y = yOf(d.tok[i] / d.base * 100);
        ectx.beginPath();
        ectx.arc(x, y, i === hoverIdx ? 7 : 5, 0, Math.PI * 2);
        ectx.fillStyle = i === hoverIdx ? '#f0f921' : d.color;
        ectx.fill();
      });
    }

    drawEE();
    document.addEventListener('themechange', drawEE);
    eeCanvas.addEventListener('pointermove', function (e) {
      var d = EE[model];
      var rect = eeCanvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) / rect.width * EW;
      var best = -1, bd = 1e9;
      d.th.forEach(function (th, i) {
        var dist = Math.abs(xOf(th) - mx);
        if (dist < bd) { bd = dist; best = i; }
      });
      if (best !== hoverIdx && bd < 60) {
        hoverIdx = best;
        drawEE();
        var tip = document.getElementById('eeTip');
        tip.textContent = '阈值 ' + d.th[best] + ' → 平均 ' + d.tok[best] + ' token（基线 ' + d.base +
          ' 的 ' + Math.round(d.tok[best] / d.base * 100) + '%），内容保留 ' + Math.round(d.kept[best] * 100) + '%';
      }
    });
    eeCanvas.addEventListener('pointerleave', function () {
      hoverIdx = -1; drawEE();
      document.getElementById('eeTip').textContent = '悬停查看每个阈值的取舍';
    });
    document.querySelectorAll('.pill').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.pill').forEach(function (b) {
          b.classList.remove('active'); b.setAttribute('aria-selected', 'false');
        });
        btn.classList.add('active'); btn.setAttribute('aria-selected', 'true');
        model = btn.getAttribute('data-model');
        hoverIdx = -1;
        drawEE();
        document.getElementById('eeTip').textContent = '悬停查看每个阈值的取舍';
      });
    });
  }

  /* ================= pip copy ================= */
  var pip = document.getElementById('pipBtn');
  if (pip) {
    pip.addEventListener('click', function () {
      var done = function () {
        var old = pip.textContent;
        pip.textContent = '已复制 ✓';
        setTimeout(function () { pip.textContent = old; }, 1400);
      };
      if (navigator.clipboard) {
        navigator.clipboard.writeText('pip install univis').then(done, done);
      } else { done(); }
    });
  }

  /* ================= scroll reveals ================= */
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
})();
