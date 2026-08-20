/* UniVis gallery interactions: hero telemetry replay, report cards,
   early-exit chart, scroll reveals. No dependencies. */

(function () {
  'use strict';

  /* ---------- plasma colormap (matplotlib plasma approximation) ---------- */
  var STOPS = [
    [0.00, 13, 8, 135],
    [0.25, 126, 3, 168],
    [0.50, 204, 71, 120],
    [0.75, 248, 148, 65],
    [1.00, 240, 249, 33],
  ];
  function plasma(t) {
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i];
        var f = (t - a[0]) / (b[0] - a[0]);
        var rgb = [0, 1, 2].map(function (k) {
          return Math.round(a[k + 1] + f * (b[k + 1] - a[k + 1]));
        });
        return 'rgb(' + rgb.join(',') + ')';
      }
    }
    return 'rgb(240,249,33)';
  }

  /* ---------- hero: replay real telemetry ---------- */
  var canvas = document.getElementById('heroCanvas');
  if (canvas && typeof HERO !== 'undefined') {
    var ctx = canvas.getContext('2d');
    var ROWS = HERO.layers, COLS = Math.min(HERO.tokens, 478);
    var DATA = HERO.data;

    // normalize with percentile clip for contrast
    var flat = [];
    for (var s = 0; s < COLS; s++) for (var l = 0; l < ROWS; l++) flat.push(DATA[s][l]);
    flat.sort(function (a, b) { return a - b; });
    var lo = flat[Math.floor(flat.length * 0.02)];
    var hi = flat[Math.floor(flat.length * 0.98)];
    var span = Math.max(hi - lo, 1e-6);

    var W = canvas.width, H = canvas.height;
    var padL = 46, padR = 10, padT = 12, padB = 26;
    var cw = (W - padL - padR) / COLS, ch = (H - padT - padB) / ROWS;

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

    function drawUpTo(col) {
      drawGrid();
      for (var s = 0; s < col; s++) {
        for (var l = 0; l < ROWS; l++) {
          var v = (DATA[s][l] - lo) / span;
          ctx.fillStyle = plasma(v);
          ctx.fillRect(padL + s * cw, padT + l * ch, Math.max(cw - 0.4, 0.6), Math.max(ch - 0.6, 0.6));
        }
      }
      // active column glow
      if (col > 0 && col < COLS) {
        ctx.fillStyle = 'rgba(240,249,33,0.16)';
        ctx.fillRect(padL + (col - 1) * cw, padT, cw * 2.2, ROWS * ch);
      }
      var lbl = document.getElementById('roToken');
      if (lbl) lbl.textContent = 'token ' + String(Math.min(col, COLS)).padStart(3, '0') + '/' + COLS;
    }

    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      drawUpTo(COLS);
      var st = document.getElementById('roState');
      if (st) st.textContent = 'static render';
    } else {
      var col = 0, last = 0, paused = false, holdUntil = 0;
      function frame(ts) {
        if (!paused) {
          if (ts > holdUntil) {
            var speed = Math.max(1, Math.round(COLS / 900)); // full pass ~9s
            drawUpTo(col);
            col += speed;
            if (col > COLS + 60) { col = 0; holdUntil = ts + 400; }
            last = ts;
          }
        }
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);

      canvas.addEventListener('pointermove', function (e) {
        paused = true;
        var rect = canvas.getBoundingClientRect();
        var x = (e.clientX - rect.left) / rect.width * W;
        var c = Math.round((x - padL) / cw);
        drawUpTo(Math.max(0, Math.min(c, COLS)));
        var st = document.getElementById('roState');
        if (st) st.textContent = 'scrub — pointer';
      });
      canvas.addEventListener('pointerleave', function () {
        paused = false;
        col = Math.max(col, 1);
        var st = document.getElementById('roState');
        if (st) st.textContent = 'replaying…';
      });
    }
  }

  /* ---------- report gallery cards ---------- */
  var REPORTS = [
    { f: 'c500_phase_a.html', hw: 'c500', m: 'Qwen2.5-0.5B', note: '阶段 A 通路验证 · 3 提示词 × 128 token · 24 层 × 384 步 · 零代码修改' },
    { f: 'c500_0.5B.html', hw: 'c500', m: 'Qwen2.5-0.5B', note: '阶段 B 基线协议 · 50 token · 与 L20 同题对照' },
    { f: 'l20_0.5B.html', hw: 'l20', m: 'Qwen2.5-0.5B', note: '阶段 B 基线协议 · 50 token · r = 1.0000' },
    { f: 'c500_3B.html', hw: 'c500', m: 'Qwen2.5-3B', note: '36 层 · 6.3GB bf16 · 与 L20 画像一致' },
    { f: 'l20_3B.html', hw: 'l20', m: 'Qwen2.5-3B', note: '36 层 · r = 1.0000 · MAE 0.0007' },
    { f: 'c500_7B.html', hw: 'c500', m: 'Qwen2.5-7B', note: '15.3GB bf16 跑通 16GB 切分实例 · 28 层' },
    { f: 'l20_7B.html', hw: 'l20', m: 'Qwen2.5-7B', note: 'r = 1.0000 · MAE 0.0005（三规模最紧）' },
    { f: 'c500_0.5B_multi.html', hw: 'c500', m: 'Qwen2.5-0.5B · 10 提示词', note: '多提示词协议 · 约 500 步聚合' },
    { f: 'l20_0.5B_multi.html', hw: 'l20', m: 'Qwen2.5-0.5B · 10 提示词', note: '多提示词协议 · r = 1.0000' },
    { f: 'c500_tinyllama.html', hw: 'c500', m: 'TinyLlama-1.1B', note: 'Llama 架构跨架构检验 · 22 层 · r = 0.9968' },
    { f: 'l20_tinyllama.html', hw: 'l20', m: 'TinyLlama-1.1B', note: 'Llama 架构 · 与 C500 侧对照' },
  ];
  var grid = document.getElementById('reportGrid');
  if (grid) {
    REPORTS.forEach(function (r) {
      var a = document.createElement('a');
      a.className = 'rcard';
      a.href = 'assets/reports/' + r.f;
      a.target = '_blank';
      a.rel = 'noopener';
      a.innerHTML =
        '<span class="tag ' + r.hw + '">' + (r.hw === 'c500' ? 'MetaX C500 · MXMACA' : 'NVIDIA L20') + '</span>' +
        '<h3>' + r.m + '</h3>' +
        '<p>' + r.note + '</p>' +
        '<span class="spark" aria-hidden="true"><i style="width:' +
        (60 + Math.random() * 40).toFixed(0) + '%;background:linear-gradient(90deg,#7e03a8,#cc4778,#f89441)"></i></span>' +
        '<span class="open">打开完整报告 ↗</span>';
      grid.appendChild(a);
    });
  }

  /* ---------- early-exit interactive chart ---------- */
  var EE = {
    q05: {
      base: 101.3,
      th: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
      tok: [99.4, 95.3, 78.9, 73.0, 47.5, 21.9],
      kept: [0.96, 0.93, 0.79, 0.74, 0.54, 0.30],
      color: '#f89441',
    },
    q3: {
      base: 102.0,
      th: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
      tok: [79.8, 70.3, 64.5, 47.4, 24.5, 8.5],
      kept: [0.79, 0.72, 0.67, 0.53, 0.33, 0.17],
      color: '#cc4778',
    },
  };
  var eeCanvas = document.getElementById('eeChart');
  if (eeCanvas) {
    var ectx = eeCanvas.getContext('2d');
    var EW = eeCanvas.width, EH = eeCanvas.height;
    var epad = { l: 56, r: 20, t: 18, b: 44 };
    var model = 'q05';
    var hoverIdx = -1;

    function xOf(th) {
      var lo = Math.log(0.05), hi = Math.log(2.0);
      return epad.l + ((Math.log(th) - lo) / (hi - lo)) * (EW - epad.l - epad.r);
    }
    function yOf(pct) {
      return EH - epad.b - (pct / 100) * (EH - epad.t - epad.b);
    }

    function drawEE() {
      var d = EE[model];
      ectx.fillStyle = '#0d1017';
      ectx.fillRect(0, 0, EW, EH);
      ectx.strokeStyle = 'rgba(255,255,255,0.10)';
      ectx.fillStyle = 'rgba(152,161,179,0.8)';
      ectx.font = '11px JetBrains Mono, monospace';
      // y grid 0..100
      for (var p = 0; p <= 100; p += 25) {
        var y = yOf(p);
        ectx.beginPath(); ectx.moveTo(epad.l, y); ectx.lineTo(EW - epad.r, y); ectx.stroke();
        ectx.fillText(p + '%', 14, y + 4);
      }
      // x ticks
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

      // baseline
      ectx.strokeStyle = 'rgba(255,255,255,0.25)';
      ectx.setLineDash([4, 4]);
      ectx.beginPath(); ectx.moveTo(epad.l, yOf(100)); ectx.lineTo(EW - epad.r, yOf(100)); ectx.stroke();
      ectx.setLineDash([]);

      // curve
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

  /* ---------- pip copy ---------- */
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

  /* ---------- scroll reveals ---------- */
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
})();
