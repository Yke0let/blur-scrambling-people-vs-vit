/* ==========================================================================
   Blur and Scrambling: People vs. Vision Models — Interactive Slide Engine
   100% real verified metrics from human_results.csv, comparison_results.csv,
   human_results_by_group.csv, all_human_responses.csv, model_predictions.csv
   ========================================================================== */

(() => {
  'use strict';

  // ---------------- VERIFIED EXPERIMENTAL DATA ----------------
  const DATA = {
    condsBlur: ['intact', 'blur_low', 'blur_medium', 'blur_high'],
    labelsBlur: ['Intact', 'σ = 2 px', 'σ = 5 px', 'σ = 10 px'],
    condsShuf: ['intact', 'shuffle_low', 'shuffle_medium', 'shuffle_high'],
    labelsShuf: ['Intact', '4 tiles (2×2)', '16 tiles (4×4)', '64 tiles (8×8)'],
    allConds: ['intact', 'blur_low', 'blur_medium', 'blur_high', 'shuffle_low', 'shuffle_medium', 'shuffle_high'],
    allCondLabels: ['Intact', 'Blur σ=2', 'Blur σ=5', 'Blur σ=10', 'Shuffle 4', 'Shuffle 16', 'Shuffle 64'],

    // Matched 70 human_uids (results_h) & Pooled Humans (N=21, 1470 trials)
    human: {
      intact:         { acc: 88.10, lo: 83.80, hi: 92.38 },
      blur_low:       { acc: 89.05, lo: 84.76, hi: 93.33 },
      blur_medium:    { acc: 82.38, lo: 77.14, hi: 87.14 },
      blur_high:      { acc: 82.38, lo: 77.14, hi: 87.62 },
      shuffle_low:    { acc: 47.14, lo: 40.48, hi: 53.81 },
      shuffle_medium: { acc: 22.38, lo: 16.67, hi: 27.62 },
      shuffle_high:   { acc: 15.24, lo: 10.48, hi: 20.48 }
    },
    clip70: {
      intact:         { acc: 98.57, lo: 95.71, hi: 100.00 },
      blur_low:       { acc: 92.86, lo: 85.71, hi: 98.57 },
      blur_medium:    { acc: 88.57, lo: 80.00, hi: 95.71 },
      blur_high:      { acc: 52.86, lo: 41.43, hi: 64.29 },
      shuffle_low:    { acc: 54.29, lo: 42.86, hi: 65.71 },
      shuffle_medium: { acc: 21.43, lo: 12.86, hi: 31.43 },
      shuffle_high:   { acc: 17.14, lo: 8.57,  hi: 25.71 }
    },
    dino70: {
      intact:         { acc: 95.71, lo: 90.00, hi: 100.00 },
      blur_low:       { acc: 97.14, lo: 92.86, hi: 100.00 },
      blur_medium:    { acc: 85.71, lo: 77.14, hi: 92.86 },
      blur_high:      { acc: 54.29, lo: 42.86, hi: 65.71 },
      shuffle_low:    { acc: 67.14, lo: 55.71, hi: 78.57 },
      shuffle_medium: { acc: 15.71, lo: 7.14,  hi: 24.29 },
      shuffle_high:   { acc: 10.00, lo: 4.25,  hi: 17.14 }
    },
    clip1000: {
      intact:         { acc: 97.7, lo: 96.7, hi: 98.6 },
      blur_low:       { acc: 91.9, lo: 90.1, hi: 93.6 },
      blur_medium:    { acc: 84.5, lo: 82.3, hi: 86.8 },
      blur_high:      { acc: 50.8, lo: 47.6, hi: 53.8 },
      shuffle_low:    { acc: 63.5, lo: 60.7, hi: 66.5 },
      shuffle_medium: { acc: 23.8, lo: 21.2, hi: 26.4 },
      shuffle_high:   { acc: 15.5, lo: 13.2, hi: 17.7 }
    },
    dino1000: {
      intact:         { acc: 97.5, lo: 96.5, hi: 98.4 },
      blur_low:       { acc: 97.5, lo: 96.5, hi: 98.4 },
      blur_medium:    { acc: 76.5, lo: 73.9, hi: 79.1 },
      blur_high:      { acc: 58.2, lo: 55.1, hi: 61.1 },
      shuffle_low:    { acc: 76.8, lo: 74.2, hi: 79.5 },
      shuffle_medium: { acc: 21.2, lo: 18.7, hi: 23.8 },
      shuffle_high:   { acc: 10.4, lo: 8.6,  hi: 12.3 }
    },

    // 3 Independent Human Replication Groups (7 participants / 490 trials each)
    groups: {
      G1: { intact: 91.43, blur_low: 85.71, blur_medium: 88.57, blur_high: 80.00, shuffle_low: 38.57, shuffle_medium: 15.71, shuffle_high: 15.71 },
      G2: { intact: 81.43, blur_low: 88.57, blur_medium: 77.14, blur_high: 85.71, shuffle_low: 42.86, shuffle_medium: 25.71, shuffle_high: 8.57 },
      G3: { intact: 91.43, blur_low: 92.86, blur_medium: 81.43, blur_high: 81.43, shuffle_low: 60.00, shuffle_medium: 25.71, shuffle_high: 21.43 }
    },

    // Per-Category Accuracy (%) under blur_high (sigma = 10 px)
    categories: ['Airplane', 'Bicycle', 'Car', 'Cat', 'Chair', 'Clock', 'Fish', 'House', 'Tree', 'Umbrella'],
    blurHighCat: {
      human:  [85.7, 90.5, 81.0, 61.9, 90.5, 76.2, 90.5, 100.0, 57.1, 90.5],
      clip70: [14.3, 28.6, 85.7, 14.3, 14.3, 100.0, 85.7, 14.3, 85.7, 85.7],
      dino70: [85.7, 42.9, 100.0, 28.6, 85.7, 42.9, 0.0, 85.7, 28.6, 42.9]
    },

    // Human Reaction Time (ms) — Median & IQR (25th - 75th percentiles) across 1470 trials
    rt: {
      median: [765, 798, 942, 943, 1410, 1883, 1630],
      q25:    [520, 539, 581, 570, 792,  963,  779],
      q75:    [1397, 1277, 1987, 1839, 2956, 3042, 3058]
    },

    // Prediction share (%) across the 10 classes under shuffle_high & blur_high
    predShareShuffleHigh: {
      dino:  [0.3, 0.1, 0.5, 0.0, 97.0, 0.0, 0.0, 0.0, 0.2, 1.9],
      clip:  [8.0, 0.0, 0.3, 0.0, 65.8, 0.0, 16.8, 0.1, 8.2, 0.8],
      human: [6.7, 10.0, 9.0, 18.6, 7.1, 6.7, 2.9, 14.8, 20.5, 3.8]
    },
    predShareBlurHigh: {
      clip:  [4.1, 1.7, 8.5, 2.3, 0.6, 33.2, 14.1, 1.2, 28.3, 6.0],
      dino:  [32.5, 7.7, 22.8, 4.0, 10.8, 3.5, 0.3, 6.4, 4.7, 7.3],
      human: [11.9, 11.0, 8.1, 10.5, 9.5, 7.6, 11.9, 10.5, 6.7, 12.4]
    },

    // Ink-Tile Fraction Split on shuffle_high (64 tiles; median = 0.438)
    inkSplit: {
      labels: ['CLIP ViT-L/14 (N=1000)', 'DINOv2 ViT-B/14 (N=1000)', 'Humans (N=21)'],
      fewInk:  [21.1, 15.1, 13.7],
      manyInk: [9.8,  5.6,  16.7]
    }
  };

  // ---------------- CUSTOM CHART.JS PLUGINS ----------------
  const errorBarsPlugin = {
    id: 'errorBarsPlugin',
    afterDatasetsDraw(chart) {
      const { ctx, scales: { y } } = chart;
      if (!y) return;
      ctx.save();
      chart.data.datasets.forEach((ds, dsIdx) => {
        if (!ds.ciLow || !ds.ciHigh) return;
        const meta = chart.getDatasetMeta(dsIdx);
        if (meta.hidden) return;
        ctx.strokeStyle = ds.errorBarColor || ds.borderColor || '#0f172a';
        ctx.lineWidth = ds.errorBarWidth || 1.7;
        meta.data.forEach((pt, i) => {
          const lo = ds.ciLow[i];
          const hi = ds.ciHigh[i];
          if (lo == null || hi == null) return;
          const x = pt.x;
          const yLo = y.getPixelForValue(lo);
          const yHi = y.getPixelForValue(hi);
          ctx.beginPath();
          ctx.moveTo(x, yLo);
          ctx.lineTo(x, yHi);
          ctx.moveTo(x - 4, yLo);
          ctx.lineTo(x + 4, yLo);
          ctx.moveTo(x - 4, yHi);
          ctx.lineTo(x + 4, yHi);
          ctx.stroke();
        });
      });
      ctx.restore();
    }
  };

  const chanceLinePlugin = {
    id: 'chanceLinePlugin',
    afterDatasetsDraw(chart) {
      const cfg = chart.options.plugins && chart.options.plugins.chanceLine;
      if (!cfg || !cfg.enabled) return;
      const { ctx, chartArea, scales: { y } } = chart;
      const yPos = y.getPixelForValue(cfg.value ?? 10);
      const label = cfg.label || 'Chance (10%)';
      const align = cfg.align || 'left';
      ctx.save();
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(chartArea.left, yPos);
      ctx.lineTo(chartArea.right, yPos);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = '700 10.5px Inter, sans-serif';
      const textW = ctx.measureText(label).width;
      const padX = 6;
      const pillH = 16;
      const pillW = textW + padX * 2;
      const pillX = align === 'right' ? (chartArea.right - pillW - 6) : (chartArea.left + 8);
      const pillY = yPos - pillH - 4;

      ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(pillX, pillY, pillW, pillH, 4);
      } else {
        ctx.rect(pillX, pillY, pillW, pillH);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#475569';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, pillX + padX, pillY + pillH / 2 + 0.5);
      ctx.restore();
    }
  };

  if (typeof Chart !== 'undefined') {
    Chart.register(errorBarsPlugin, chanceLinePlugin);
    Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
    Chart.defaults.color = '#334155';
  }

  // ---------------- CHART MANAGEMENT ----------------
  const activeCharts = {};

  function makeChart(canvasId, config) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === 'undefined') return null;
    if (activeCharts[canvasId]) {
      activeCharts[canvasId].destroy();
      delete activeCharts[canvasId];
    }
    const ch = new Chart(el.getContext('2d'), config);
    activeCharts[canvasId] = ch;
    return ch;
  }

  function series(obj, conds) {
    return {
      data: conds.map(c => obj[c].acc),
      ciLow: conds.map(c => obj[c].lo),
      ciHigh: conds.map(c => obj[c].hi)
    };
  }

  const chartInitializers = {
    // Slide 12: Main Finding — Two-panel (Blur | Tile Shuffle) on matched 70 human_uids vs Humans
    12(instant = false) {
      const anim = instant ? false : { duration: 1050, easing: 'easeOutQuart' };
      const buildPanel = (canvasId, conds, labels) => {
        const h = series(DATA.human, conds);
        const c = series(DATA.clip70, conds);
        const d = series(DATA.dino70, conds);
        makeChart(canvasId, {
          type: 'line',
          data: {
            labels,
            datasets: [
              {
                label: 'Humans (N=21, 1470 trials)',
                data: h.data, ciLow: h.ciLow, ciHigh: h.ciHigh,
                borderColor: '#dc2626', backgroundColor: '#dc2626',
                borderWidth: 3.2, borderDash: [6, 4],
                pointStyle: 'triangle', pointRadius: 7, pointHoverRadius: 9,
                tension: 0.14
              },
              {
                label: 'CLIP ViT-L/14 (70 human UIDs)',
                data: c.data, ciLow: c.ciLow, ciHigh: c.ciHigh,
                borderColor: '#2563eb', backgroundColor: '#2563eb',
                borderWidth: 2.6, pointStyle: 'circle', pointRadius: 5.5, pointHoverRadius: 8,
                tension: 0.14
              },
              {
                label: 'DINOv2 ViT-B/14 (70 human UIDs)',
                data: d.data, ciLow: d.ciLow, ciHigh: d.ciHigh,
                borderColor: '#059669', backgroundColor: '#059669',
                borderWidth: 2.6, pointStyle: 'rect', pointRadius: 5.5, pointHoverRadius: 8,
                tension: 0.14
              }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: anim,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, padding: 14, font: { size: 11.5, weight: '600' } } },
              chanceLine: { enabled: true, value: 10, label: 'Chance (10%)' },
              tooltip: {
                callbacks: {
                  label: ctx => {
                    const ds = ctx.dataset;
                    const i = ctx.dataIndex;
                    return `${ds.label}: ${ctx.parsed.y.toFixed(1)}% [95% CI: ${ds.ciLow[i].toFixed(1)}–${ds.ciHigh[i].toFixed(1)}%]`;
                  }
                }
              }
            },
            scales: {
              y: { min: 0, max: 105, title: { display: true, text: '10-way accuracy (%)', font: { weight: '700', size: 11.5 } }, grid: { color: '#f1f5f9' } },
              x: { grid: { display: false }, ticks: { font: { weight: '600', size: 11.5 } } }
            }
          }
        });
      };
      buildPanel('chart-main-blur', DATA.condsBlur, DATA.labelsBlur);
      buildPanel('chart-main-shuffle', DATA.condsShuf, DATA.labelsShuf);
    },

    // Slide 15: Per-Category Accuracy (%) under blur_high (sigma = 10 px)
    15(instant = false) {
      const anim = instant ? false : { duration: 1000, easing: 'easeOutQuart' };
      makeChart('chart-cat-blur', {
        type: 'bar',
        data: {
          labels: DATA.categories,
          datasets: [
            { label: 'Humans (N=21)', data: DATA.blurHighCat.human, backgroundColor: '#dc2626', borderRadius: 4 },
            { label: 'CLIP ViT-L/14 (70 UIDs)', data: DATA.blurHighCat.clip70, backgroundColor: '#2563eb', borderRadius: 4 },
            { label: 'DINOv2 ViT-B/14 (70 UIDs)', data: DATA.blurHighCat.dino70, backgroundColor: '#059669', borderRadius: 4 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12, padding: 16, font: { size: 12, weight: '600' } } },
            chanceLine: { enabled: true, value: 10, label: 'Chance (10%)' },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` } }
          },
          scales: {
            y: { min: 0, max: 108, title: { display: true, text: 'Accuracy at blur_high (σ = 10 px) (%)', font: { weight: '700' } }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false }, ticks: { font: { weight: '600', size: 11.5 } } }
          }
        }
      });
    },

    // Slide 16: Ruling out boring explanations — (A) 70 UIDs vs N=1000 | (B) 3 Human Groups Replication
    16(instant = false) {
      const anim = instant ? false : { duration: 980, easing: 'easeOutQuart' };
      const c70 = DATA.allConds.map(c => DATA.clip70[c].acc);
      const c1k = DATA.allConds.map(c => DATA.clip1000[c].acc);
      const d70 = DATA.allConds.map(c => DATA.dino70[c].acc);
      const d1k = DATA.allConds.map(c => DATA.dino1000[c].acc);

      makeChart('chart-ctrl-subset', {
        type: 'line',
        data: {
          labels: DATA.allCondLabels,
          datasets: [
            { label: 'CLIP (N=1000)', data: c1k, borderColor: '#2563eb', backgroundColor: '#2563eb', borderWidth: 2.4, pointRadius: 4, tension: 0.12 },
            { label: 'CLIP (70 UIDs)', data: c70, borderColor: '#93c5fd', backgroundColor: '#ffffff', borderWidth: 2.2, borderDash: [5, 4], pointRadius: 4.5, pointBorderColor: '#2563eb', tension: 0.12 },
            { label: 'DINOv2 (N=1000)', data: d1k, borderColor: '#059669', backgroundColor: '#059669', borderWidth: 2.4, pointRadius: 4, tension: 0.12 },
            { label: 'DINOv2 (70 UIDs)', data: d70, borderColor: '#6ee7b7', backgroundColor: '#ffffff', borderWidth: 2.2, borderDash: [5, 4], pointRadius: 4.5, pointBorderColor: '#059669', tension: 0.12 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 10, padding: 10, font: { size: 10.5, weight: '600' } } },
            chanceLine: { enabled: true, value: 10 }
          },
          scales: {
            y: { min: 0, max: 105, title: { display: true, text: 'Accuracy (%)', font: { weight: '700', size: 11 } }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false }, ticks: { font: { size: 10, weight: '600' } } }
          }
        }
      });

      makeChart('chart-ctrl-groups', {
        type: 'line',
        data: {
          labels: DATA.allCondLabels,
          datasets: [
            { label: 'Group 1 (P01–P07)', data: DATA.allConds.map(c => DATA.groups.G1[c]), borderColor: '#2563eb', backgroundColor: '#2563eb', borderWidth: 2.4, pointRadius: 4.5, tension: 0.14 },
            { label: 'Group 2 (P08–P14)', data: DATA.allConds.map(c => DATA.groups.G2[c]), borderColor: '#059669', backgroundColor: '#059669', borderWidth: 2.4, pointRadius: 4.5, tension: 0.14 },
            { label: 'Group 3 (P15–P21)', data: DATA.allConds.map(c => DATA.groups.G3[c]), borderColor: '#d97706', backgroundColor: '#d97706', borderWidth: 2.4, pointRadius: 4.5, tension: 0.14 },
            { label: 'Pooled Humans (N=21)', data: DATA.allConds.map(c => DATA.human[c].acc), borderColor: '#dc2626', backgroundColor: '#dc2626', borderWidth: 3, borderDash: [5, 4], pointRadius: 5, pointStyle: 'triangle', tension: 0.14 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 10, padding: 10, font: { size: 10.5, weight: '600' } } },
            chanceLine: { enabled: true, value: 10 }
          },
          scales: {
            y: { min: 0, max: 105, title: { display: true, text: 'Human Accuracy (%)', font: { weight: '700', size: 11 } }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false }, ticks: { font: { size: 10, weight: '600' } } }
          }
        }
      });
    },

    // Slide 17: Human Reaction Time (Median + IQR whiskers)
    17(instant = false) {
      const anim = instant ? false : { duration: 1000, easing: 'easeOutQuart' };
      makeChart('chart-rt', {
        type: 'bar',
        data: {
          labels: DATA.allCondLabels,
          datasets: [{
            label: 'Median Reaction Time (ms) with IQR (25th–75th pct)',
            data: DATA.rt.median,
            ciLow: DATA.rt.q25,
            ciHigh: DATA.rt.q75,
            errorBarColor: '#0f172a',
            errorBarWidth: 2.0,
            backgroundColor: ['#334155', '#60a5fa', '#3b82f6', '#2563eb', '#a78bfa', '#7c3aed', '#6d28d9'],
            borderRadius: 6,
            barPercentage: 0.56
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'top', labels: { font: { size: 11.5, weight: '600' } } },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const i = ctx.dataIndex;
                  return `Median: ${DATA.rt.median[i]} ms (IQR: ${DATA.rt.q25[i]}–${DATA.rt.q75[i]} ms)`;
                }
              }
            }
          },
          scales: {
            y: { min: 0, max: 3400, title: { display: true, text: 'Response Latency (ms)', font: { weight: '700' } }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false }, ticks: { font: { weight: '700', size: 11.5 } } }
          }
        }
      });
    },

    // Slide 18: Failure Mode 1 — Degenerate Class-Collapse under shuffle_high & blur_high
    18(instant = false) {
      const anim = instant ? false : { duration: 1050, easing: 'easeOutQuart' };
      makeChart('chart-collapse-shuf', {
        type: 'bar',
        data: {
          labels: DATA.categories,
          datasets: [
            { label: 'DINOv2 ViT-B/14 (97.0% chair!)', data: DATA.predShareShuffleHigh.dino, backgroundColor: '#059669', borderRadius: 3 },
            { label: 'CLIP ViT-L/14 (65.8% chair!)', data: DATA.predShareShuffleHigh.clip, backgroundColor: '#2563eb', borderRadius: 3 },
            { label: 'Humans (spread across classes)', data: DATA.predShareShuffleHigh.human, backgroundColor: '#dc2626', borderRadius: 3 }
          ]
        },
        options: {
          indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 10, padding: 10, font: { size: 10.5, weight: '600' } } },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)}% of all predictions` } }
          },
          scales: {
            x: { min: 0, max: 100, title: { display: true, text: 'Share of all predictions (%) — ideal uniform prior = 10%', font: { size: 10.5, weight: '700' } }, grid: { color: '#f1f5f9' } },
            y: { grid: { display: false }, ticks: { font: { size: 10.5, weight: '600' } } }
          }
        }
      });

      makeChart('chart-collapse-blur', {
        type: 'bar',
        data: {
          labels: DATA.categories,
          datasets: [
            { label: 'CLIP (61.5% clock/tree)', data: DATA.predShareBlurHigh.clip, backgroundColor: '#2563eb', borderRadius: 3 },
            { label: 'DINOv2 (55.3% airplane/car)', data: DATA.predShareBlurHigh.dino, backgroundColor: '#059669', borderRadius: 3 },
            { label: 'Humans (6.7–12.4%, well-calibrated)', data: DATA.predShareBlurHigh.human, backgroundColor: '#dc2626', borderRadius: 3 }
          ]
        },
        options: {
          indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 10, padding: 10, font: { size: 10.5, weight: '600' } } },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)}% of all predictions` } }
          },
          scales: {
            x: { min: 0, max: 40, title: { display: true, text: 'Share of all predictions (%) — ideal uniform prior = 10%', font: { size: 10.5, weight: '700' } }, grid: { color: '#f1f5f9' } },
            y: { grid: { display: false }, ticks: { font: { size: 10.5, weight: '600' } } }
          }
        }
      });
    },

    // Slide 19: Failure Modes 2 & 3 — Ink-Tile Sparsity Split on shuffle_high
    19(instant = false) {
      const anim = instant ? false : { duration: 1000, easing: 'easeOutQuart' };
      makeChart('chart-ink-split', {
        type: 'bar',
        data: {
          labels: DATA.inkSplit.labels,
          datasets: [
            { label: 'Few Ink Tiles (≤ 43.8% median)', data: DATA.inkSplit.fewInk, backgroundColor: '#64748b', borderRadius: 5 },
            { label: 'Many Ink Tiles (> 43.8% median)', data: DATA.inkSplit.manyInk, backgroundColor: '#dc2626', borderRadius: 5 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: anim,
          plugins: {
            legend: { position: 'bottom', labels: { boxWidth: 11, padding: 12, font: { size: 11, weight: '600' } } },
            chanceLine: { enabled: true, value: 10, label: 'Chance (10%)' },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` } }
          },
          scales: {
            y: { min: 0, max: 30, title: { display: true, text: 'shuffle_high (64 tiles) Acc (%)', font: { weight: '700', size: 11 } }, grid: { color: '#f1f5f9' } },
            x: { grid: { display: false }, ticks: { font: { weight: '600', size: 11 } } }
          }
        }
      });
    }
  };

  // ---------------- NAVIGATION, HASH ROUTING, SECBAR & MODALS ----------------
  const INSTANT_RENDER = new URLSearchParams(location.search).get('static') === '1';
  if (INSTANT_RENDER) document.body.classList.add('static-render');

  const slides = Array.from(document.querySelectorAll('.slide'));
  const totalSlides = slides.length;
  let currentIdx = 0;

  // Build Segmented Section Bar (#secbar) excluding Cover
  const secOrder = [];
  slides.forEach((s, idx) => {
    const sec = s.dataset.section || '';
    if (!sec || sec === 'Cover') return;
    let found = secOrder.find(x => x.name === sec);
    if (!found) {
      found = { name: sec, indices: [] };
      secOrder.push(found);
    }
    found.indices.push(idx);
  });

  const secbarEl = document.getElementById('secbar');
  secOrder.forEach(sec => {
    const btn = document.createElement('button');
    btn.className = 'sec-seg';
    btn.style.flexGrow = String(sec.indices.length);
    btn.textContent = sec.name;
    btn.title = `Jump to ${sec.name}`;
    btn.addEventListener('click', () => goToSlide(sec.indices[0]));
    secbarEl.appendChild(btn);
    sec.el = btn;
  });

  // Build Overview Grid (#overview-grid)
  const overviewGrid = document.getElementById('overview-grid');
  slides.forEach((s, idx) => {
    const item = document.createElement('button');
    item.className = 'ov-item';
    item.dataset.idx = String(idx);
    const title = s.dataset.title || `Slide ${idx + 1}`;
    const sec = s.dataset.section || 'Slide';
    item.innerHTML = `<span class="ov-num">#${String(idx + 1).padStart(2, '0')}</span><span class="ov-sec">${sec}</span><span class="ov-title">${title}</span>`;
    item.addEventListener('click', () => {
      toggleOverview(false);
      goToSlide(idx);
    });
    overviewGrid.appendChild(item);
  });

  function updateSectionBar(idx) {
    const secName = slides[idx].dataset.section || '';
    if (!secName || secName === 'Cover') {
      secbarEl.classList.add('hidden-bar');
      return;
    }
    secbarEl.classList.remove('hidden-bar');
    const activeSecPos = secOrder.findIndex(s => s.name === secName);
    secOrder.forEach((sec, pos) => {
      sec.el.classList.toggle('done', pos < activeSecPos);
      sec.el.classList.toggle('active', pos === activeSecPos);
      if (pos === activeSecPos) {
        const localPos = sec.indices.indexOf(idx);
        const pct = (localPos + 1) / sec.indices.length;
        sec.el.style.setProperty('--p', pct.toFixed(3));
      } else {
        sec.el.style.removeProperty('--p');
      }
    });
  }

  function updateNotes(idx) {
    const notesText = document.getElementById('notes-text');
    const notesTitle = document.getElementById('notes-slide-title');
    if (notesText) notesText.textContent = slides[idx].dataset.notes || '';
    if (notesTitle) notesTitle.textContent = `Slide ${idx + 1} / ${totalSlides} — ${slides[idx].dataset.title || ''}`;
  }

  function goToSlide(idx, updateHash = true) {
    const clamped = Math.max(0, Math.min(totalSlides - 1, idx));
    currentIdx = clamped;

    slides.forEach((s, i) => {
      if (i === clamped) {
        s.classList.remove('anim-ready');
        s.classList.add('active');
        if (INSTANT_RENDER) {
          s.classList.add('anim-ready');
        } else {
          // Force reflow so animations replay cleanly on every entry
          void s.offsetWidth;
          requestAnimationFrame(() => s.classList.add('anim-ready'));
        }
      } else {
        s.classList.remove('active', 'anim-ready');
      }
    });

    const slideNumber = clamped + 1;
    document.getElementById('hud-counter').textContent = `${String(slideNumber).padStart(2, '0')} / ${String(totalSlides).padStart(2, '0')}`;
    document.getElementById('hud-sec-label').textContent = slides[clamped].dataset.section || '';

    updateSectionBar(clamped);
    updateNotes(clamped);

    document.querySelectorAll('.ov-item').forEach((el, i) => {
      el.classList.toggle('current', i === clamped);
    });

    if (updateHash) {
      try {
        history.replaceState(null, '', `${location.pathname}${location.search}#${slideNumber}`);
      } catch (_) {}
    }

    // Trigger dynamic Chart.js rendering with fresh animation on chart slides
    if (chartInitializers[slideNumber]) {
      if (INSTANT_RENDER) {
        chartInitializers[slideNumber](true);
      } else {
        requestAnimationFrame(() => chartInitializers[slideNumber](false));
      }
    }
  }

  function nextSlide() { if (currentIdx < totalSlides - 1) goToSlide(currentIdx + 1); }
  function prevSlide() { if (currentIdx > 0) goToSlide(currentIdx - 1); }

  function toggleNotes(force) {
    const drawer = document.getElementById('notes-drawer');
    const btn = document.getElementById('btn-notes');
    const open = typeof force === 'boolean' ? force : !drawer.classList.contains('shown');
    drawer.classList.toggle('shown', open);
    if (btn) btn.classList.toggle('is-active', open);
  }

  function toggleOverview(force) {
    const modal = document.getElementById('overview-modal');
    const btn = document.getElementById('btn-overview');
    const open = typeof force === 'boolean' ? force : !modal.classList.contains('shown');
    modal.classList.toggle('shown', open);
    if (btn) btn.classList.toggle('is-active', open);
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }

  // Wire HUD buttons
  document.getElementById('btn-prev').addEventListener('click', prevSlide);
  document.getElementById('btn-next').addEventListener('click', nextSlide);
  document.getElementById('btn-notes').addEventListener('click', () => toggleNotes());
  document.getElementById('btn-overview').addEventListener('click', () => toggleOverview());
  document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);
  document.getElementById('btn-print').addEventListener('click', () => {
    renderAllChartsForPrint();
    setTimeout(() => window.print(), 120);
  });
  document.getElementById('btn-close-overview').addEventListener('click', () => toggleOverview(false));
  document.getElementById('btn-close-notes').addEventListener('click', () => toggleNotes(false));

  // Keyboard navigation
  window.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'Escape') {
      toggleOverview(false);
      toggleNotes(false);
      return;
    }
    if (document.getElementById('overview-modal').classList.contains('shown')) return;

    switch (e.key) {
      case 'ArrowRight':
      case 'PageDown':
      case ' ':
      case 'Enter':
        e.preventDefault();
        nextSlide();
        break;
      case 'ArrowLeft':
      case 'PageUp':
        e.preventDefault();
        prevSlide();
        break;
      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;
      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
      case 'n':
      case 'N':
        toggleNotes();
        break;
      case 'o':
      case 'O':
        toggleOverview();
        break;
      case 'f':
      case 'F':
        toggleFullscreen();
        break;
    }
  });

  // Touch swipe navigation
  let touchStartX = 0;
  window.addEventListener('touchstart', e => { touchStartX = e.changedTouches[0].screenX; }, { passive: true });
  window.addEventListener('touchend', e => {
    const dx = e.changedTouches[0].screenX - touchStartX;
    if (Math.abs(dx) > 60) {
      if (dx < 0) nextSlide();
      else prevSlide();
    }
  }, { passive: true });

  // Render every chart statically before browser print / PDF export
  function renderAllChartsForPrint() {
    Object.keys(chartInitializers).forEach(num => {
      chartInitializers[Number(num)](true);
    });
  }
  window.addEventListener('beforeprint', renderAllChartsForPrint);

  // Hash navigation
  function loadFromHash() {
    const m = location.hash.match(/^#(\d+)$/);
    const n = m ? parseInt(m[1], 10) : 1;
    goToSlide(n - 1, false);
  }
  window.addEventListener('hashchange', loadFromHash);
  loadFromHash();
})();
