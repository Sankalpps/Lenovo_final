/**
 * NVMe Health Simulator Logic
 * Handles real-time syncing between sliders and number inputs,
 * dynamic styling for danger thresholds, and debounced API calls.
 */

(() => {
  "use strict";

  let activeAlgo = null;
  let predictTimeout = null;

  // Colors
  const COLOR_SAFE = "#34d399";   // Green
  const COLOR_WARN = "#fbbf24";   // Amber
  const COLOR_DANGER = "#f87171"; // Red
  const COLOR_ACCENT = "#22d3ee"; // Cyan
  
  const colorMap = {
    "No Failure": "#34d399",           // Green
    "Wear-Out Failure": "#f87171",     // Red
    "Thermal Failure": "#fbbf24",      // Amber
    "Firmware Failure": "#6366f1",     // Indigo
    "Media Error Failure": "#a855f7",   // Purple
    "Unsafe Shutdown Failure": "#f97316" // Orange
  };

  function init() {
    bindInputs();
    bindPresets();
    
    // Bind Predict button
    const predictBtn = document.getElementById("btn-predict-action");
    if(predictBtn) predictBtn.addEventListener("click", triggerPrediction);
    
    // Initial fetch metadata
    fetch("/api/metadata")
      .then(res => res.json())
      .then(meta => {
        if (meta.accuracy != null) {
          document.getElementById("chip-acc").textContent = `Accuracy: ${(meta.accuracy * 100).toFixed(1)}%`;
        }
      }).catch(console.error);

    // Remove initial prediction
  }

  // ─── Input Syncing & Live Updates ────────────────────────────────
  
  function bindInputs() {
    const sliders = document.querySelectorAll('input[type="range"]');

    sliders.forEach(slider => {
      const numInput = document.getElementById(slider.id + "_val");
      // Format slider track on load
      updateSliderStyle(slider);

      // On slider drag
      slider.addEventListener("input", (e) => {
        if (numInput) numInput.value = e.target.value;
        updateSliderStyle(e.target);
      });

      // On number input typing
      if (numInput) {
        numInput.addEventListener("input", (e) => {
          let val = parseFloat(e.target.value);
          if (isNaN(val)) return;
          slider.value = val;
          updateSliderStyle(slider);
        });
      }
    });
  }

  function updateSliderStyle(slider) {
    const val = parseFloat(slider.value);
    const min = parseFloat(slider.min) || 0;
    const max = parseFloat(slider.max) || 100;
    
    // Calculate percentage fill
    const percent = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    slider.style.setProperty("--fill-percent", `${percent}%`);

    // Color logic based on data-danger threshold
    let color = COLOR_ACCENT;
    if (slider.hasAttribute("data-danger")) {
      const dangerHit = parseFloat(slider.getAttribute("data-danger"));
      const isHighDanger = val >= dangerHit;
      const isWarn = val >= dangerHit * 0.7; // warn at 70% of danger

      if (isHighDanger) color = COLOR_DANGER;
      else if (isWarn) color = COLOR_WARN;
      else color = COLOR_SAFE;
    }
    slider.style.setProperty("--fill-color", color);
  }

  // ─── Debounced API Call ──────────────────────────────────────────

  function triggerPrediction() {
    // Show loading on chip
    document.getElementById("chip-acc").innerHTML = '<span class="chip-dot" style="background:#fbbf24;box-shadow:none;"></span> Inference...';

    const data = getFormData();
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
      .then(res => res.json())
      .then(resData => {
        showResults(resData);
        // Restore chip
        document.getElementById("chip-acc").textContent = "Live Stream Active";
        
        // Also fetch independent algorithms
        fetchIndependentAlgorithms(data);
      })
      .catch(console.error);
  }

  function fetchIndependentAlgorithms(formData) {
    fetch("/api/independent-algorithms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    })
    .then(res => res.json())
    .then(data => {
      displayIndependentAlgorithms(data);
    })
    .catch(err => {
      console.error("Independent algorithms fetch error:", err);
      // Don't break the main UI if this fails
    });
  }

  function displayIndependentAlgorithms(data) {
    const card = document.getElementById("independent-card");
    const container = document.getElementById("independent-results");
    
    if (!data.independent_algorithms || data.independent_algorithms.length === 0) {
      card.style.display = "none";
      return;
    }
    
    card.style.display = "block";
    
    container.innerHTML = data.independent_algorithms.map((a, i) => {
      const score = a.score;
      let rankClass = "low";
      let barColor = COLOR_SAFE;
      if (score >= 50) { rankClass = "top"; barColor = COLOR_DANGER; }
      else if (score >= 25) { rankClass = "mid"; barColor = COLOR_WARN; }
      
      return `
        <div class="algo-row independent-row" data-idx="${i}" onclick="window.toggleIndependentDetail(${i})">
          <div class="algo-rank ${rankClass}">${i + 1}</div>
          <span class="algo-name">${a.label}</span>
          <div class="algo-score-wrap">
            <div class="algo-bar-track">
              <div class="algo-bar-fill" style="width:${score}%;background:${barColor}"></div>
            </div>
            <span class="algo-score-val" style="color:${barColor}">${score}</span>
          </div>
          <span class="filter-badge">${a.error_filter}</span>
        </div>`;
    }).join("");
    
    // Store for detail view
    window._independentResults = data.independent_algorithms;
  }

  function getFormData() {
    const data = {};
    document.querySelectorAll('input[type="range"]').forEach(s => {
      data[s.id] = parseFloat(s.value) || 0;
    });
    return data;
  }

  // ─── UI Update Actions ──────────────────────────────────────────

  function showResults(data) {
    // Verdict
    const verdict = document.getElementById("verdict");
    const vIcon = document.getElementById("verdict-icon");
    const vTitle = document.getElementById("verdict-title");
    const vSub = document.getElementById("verdict-sub");

    if (data.is_healthy) {
      verdict.className = "verdict safe";
      vIcon.innerHTML = "&#10003;";
      vTitle.textContent = "Status Optimal";
      vSub.textContent = "AI detects no anomalous patterns in telemetry stream.";
    } else {
      const topAlgo = data.algorithm_results[0];
      
      // If AI thinks it's healthy but algorithms flag a risk
      if (data.ml_prediction.mode === 0) {
          verdict.className = "verdict warning";
          vIcon.innerHTML = "&#9888;";
          vTitle.textContent = `Early Warning: ${topAlgo.label} Risk`;
          vSub.textContent = `AI detects no immediate failure, but underlying degradation detected (Score: ${topAlgo.score}).`;
      } else {
          // Both AI and rules agree on a massive failure
          verdict.className = "verdict danger";
          vIcon.innerHTML = "&#9888;";
          vTitle.textContent = `Critical Alert: ${data.ml_prediction.label}`;
          vSub.textContent = `Corroborated by ${topAlgo.label} (Score: ${topAlgo.score}). Immediate investigation required.`;
      }
    }

    // Update Insights Box
    const riskBadge = document.getElementById("risk-level-badge");
    const insightFactors = document.getElementById("insight-factors");
    const insightActions = document.getElementById("insight-actions");

    if (data.risk_level) {
      riskBadge.textContent = data.risk_level;
      riskBadge.className = `insight-value risk-badge risk-${data.risk_level.toLowerCase()}`;
    }
    
    if(data.insights) {
      insightFactors.textContent = data.insights.top_contributing_factors;
      insightActions.textContent = data.insights.suggested_actions;
    }

    // ML prediction
    const mlMode = document.getElementById("ml-mode");
    mlMode.textContent = data.ml_prediction.label;
    mlMode.className = "ml-mode " + (data.ml_prediction.mode === 0 ? "no-fail" : "fail");

    // ML probabilities
    const mlProbs = document.getElementById("ml-probs");
    const entries = Object.entries(data.ml_prediction.probabilities).sort((a, b) => b[1] - a[1]);
    mlProbs.innerHTML = entries.map(([label, val]) => {
      const pct = (val * 100).toFixed(1);
      const color = colorMap[label] || COLOR_ACCENT;
      return `
        <div class="prob-row">
          <span class="prob-label">${label}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <span class="prob-val">${pct}%</span>
        </div>`;
    }).join("");

    // Algorithm results
    const algoContainer = document.getElementById("algo-results");
    algoContainer.innerHTML = data.algorithm_results.map((a, i) => {
      const score = a.score;
      let rankClass = "low";
      let barColor = COLOR_SAFE;
      if (score >= 50) { rankClass = "top"; barColor = COLOR_DANGER; }
      else if (score >= 25) { rankClass = "mid"; barColor = COLOR_WARN; }

      return `
        <div class="algo-row" data-idx="${i}" onclick="window.toggleAlgoDetail(${i})">
          <div class="algo-rank ${rankClass}">${i + 1}</div>
          <span class="algo-name">${a.label}</span>
          <div class="algo-score-wrap">
            <div class="algo-bar-track">
              <div class="algo-bar-fill" style="width:${score}%;background:${barColor}"></div>
            </div>
            <span class="algo-score-val" style="color:${barColor}">${score}</span>
          </div>
        </div>`;
    }).join("");

    // Reattach global toggler and data so HTML onclick can reach it
    window._latestAlgorithmResults = data.algorithm_results;
  }

  // ─── Details ──────────────────────────────────────────

  window.toggleAlgoDetail = function(idx) {
    const algo = window._latestAlgorithmResults[idx];
    const panel = document.getElementById("detail-panel");
    const title = document.getElementById("detail-title");
    const reasons = document.getElementById("detail-reasons");
    const rows = document.querySelectorAll(".algo-row");

    rows.forEach(r => r.classList.remove("active"));

    if (activeAlgo === algo.label) {
      panel.classList.add("hidden");
      activeAlgo = null;
      return;
    }

    activeAlgo = algo.label;
    if (rows[idx]) rows[idx].classList.add("active");
    title.textContent = algo.label + " — Diagnostic Trace";

    if (algo.reasons.length === 0) {
      reasons.innerHTML = `<div class="reason-item"><span class="reason-icon">&#10003;</span> All checks passed for this subsystem.</div>`;
    } else {
      reasons.innerHTML = algo.reasons.map(r => 
        `<div class="reason-item"><span class="reason-icon">&#9888;</span>${r}</div>`
      ).join("");
    }
    
    panel.classList.remove("hidden");
  };

  window.toggleIndependentDetail = function(idx) {
    const algo = window._independentResults?.[idx];
    if (!algo) return;
    
    const panel = document.getElementById("detail-panel");
    const title = document.getElementById("detail-title");
    const reasons = document.getElementById("detail-reasons");
    const rows = document.querySelectorAll(".independent-row");

    rows.forEach(r => r.classList.remove("active"));

    if (activeAlgo === algo.label) {
      panel.classList.add("hidden");
      activeAlgo = null;
      return;
    }

    activeAlgo = algo.label;
    if (rows[idx]) rows[idx].classList.add("active");
    title.textContent = algo.label + " — Advanced Diagnostic";

    if (algo.reasons.length === 0) {
      reasons.innerHTML = `<div class="reason-item"><span class="reason-icon">✓</span> All checks passed for this subsystem.</div>`;
    } else {
      reasons.innerHTML = algo.reasons.map(r => 
        `<div class="reason-item"><span class="reason-icon">⚠</span>${r}</div>`
      ).join("");
    }
    
    panel.classList.remove("hidden");
  };

  // ─── Presets ──────────────────────────────────────────

  function bindPresets() {
    document.getElementById("btn-sample").addEventListener("click", () => {
      setValues({
        Power_On_Hours: 54000, Total_TBW_TB: 405, Total_TBR_TB: 376,
        Temperature_C: 62, Percent_Life_Used: 95, Media_Errors: 6,
        Unsafe_Shutdowns: 8, CRC_Errors: 3, Read_Error_Rate: 24.5,
        Write_Error_Rate: 18.2
      });
    });

    document.getElementById("btn-healthy").addEventListener("click", () => {
      setValues({
        Power_On_Hours: 500, Total_TBW_TB: 2, Total_TBR_TB: 1,
        Temperature_C: 38, Percent_Life_Used: 1, Media_Errors: 0,
        Unsafe_Shutdowns: 0, CRC_Errors: 0, Read_Error_Rate: 0,
        Write_Error_Rate: 0
      });
    });

    document.getElementById("btn-thermal").addEventListener("click", () => {
      setValues({
        Power_On_Hours: 20000, Total_TBW_TB: 80, Total_TBR_TB: 60,
        Temperature_C: 85, Percent_Life_Used: 15, Media_Errors: 0,
        Unsafe_Shutdowns: 1, CRC_Errors: 0, Read_Error_Rate: 22.5,
        Write_Error_Rate: 15.5
      });
    });
  }

  function setValues(valMap) {
    Object.entries(valMap).forEach(([k, v]) => {
      const slider = document.getElementById(k);
      if (!slider) {
        return;
      }
      slider.value = v;
      const numInput = document.getElementById(k + "_val");
      if (numInput) numInput.value = v;
      updateSliderStyle(slider);
    });
  }

  // Go
  document.addEventListener("DOMContentLoaded", init);

})();
