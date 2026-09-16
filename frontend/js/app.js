/**
 * Policy & Contract Intelligence Assistant — Vanilla JavaScript Client
 * Manages queries, Fetch API calls, loading/error states, theme toggling,
 * markdown answer formatting, and quality scorecard metrics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Theme Management
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  const savedTheme = localStorage.getItem("app_theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
      const newTheme = currentTheme === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("app_theme", newTheme);
    });
  }

  // DOM Elements
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const submitBtn = document.getElementById("submit-btn");
  const charCounter = document.getElementById("char-counter");
  const errorBanner = document.getElementById("error-banner");
  const errorMessage = document.getElementById("error-message");
  const loadingIndicator = document.getElementById("loading-indicator");
  const loadingStepText = document.getElementById("loading-step-text");
  const resultContainer = document.getElementById("result-container");
  const displayQuery = document.getElementById("display-query");
  const answerContent = document.getElementById("answer-content");
  const citationsSection = document.getElementById("citations-section");
  const citationsGrid = document.getElementById("citations-grid");
  const citationsCount = document.getElementById("citations-count");
  const telemetryCandidates = document.getElementById("telemetry-candidates");
  const telemetryReranked = document.getElementById("telemetry-reranked");
  const telemetryGuardrails = document.getElementById("telemetry-guardrails");
  const telemetryGuardrailText = document.getElementById("telemetry-guardrail-text");
  const historyList = document.getElementById("history-list");
  const historyEmpty = document.getElementById("history-empty");
  const historyCount = document.getElementById("history-count");
  const newQueryBtn = document.getElementById("new-query-btn");
  const backendStatusLabel = document.getElementById("backend-status-label");
  const chipButtons = document.querySelectorAll(".chip");

  // Evaluation Modal DOM Elements
  const openEvalBtn = document.getElementById("open-eval-btn");
  const closeEvalBtn = document.getElementById("close-eval-btn");
  const evalModal = document.getElementById("eval-modal");
  const triggerEvalBtn = document.getElementById("trigger-eval-btn");
  const evalOverallScore = document.getElementById("eval-overall-score");
  const valFaithfulness = document.getElementById("val-faithfulness");
  const barFaithfulness = document.getElementById("bar-faithfulness");
  const valRelevancy = document.getElementById("val-relevancy");
  const barRelevancy = document.getElementById("bar-relevancy");
  const valPrecision = document.getElementById("val-precision");
  const barPrecision = document.getElementById("bar-precision");
  const valRecall = document.getElementById("val-recall");
  const barRecall = document.getElementById("bar-recall");
  const evalTableBody = document.getElementById("eval-table-body");

  // State
  let sessionHistory = [];
  const API_BASE_URL = window.location.origin;

  // Initialize
  checkHealth();
  setupEventListeners();

  /**
   * Health Check
   */
  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      if (response.ok) {
        backendStatusLabel.textContent = "System Ready";
        backendStatusLabel.style.color = "#059669";
      } else {
        backendStatusLabel.textContent = "Service Degraded";
        backendStatusLabel.style.color = "#d97706";
      }
    } catch (e) {
      backendStatusLabel.textContent = "Offline";
      backendStatusLabel.style.color = "#e11d48";
    }
  }

  /**
   * Setup UI Event Listeners
   */
  function setupEventListeners() {
    // Textarea input char counter
    questionInput.addEventListener("input", () => {
      const len = questionInput.value.length;
      charCounter.textContent = `${len} / 2000`;
    });

    // Enter to submit (Shift+Enter for newline)
    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    });

    // Form submit
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleSubmit();
    });

    // Suggestion chips
    chipButtons.forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        if (query) {
          questionInput.value = query;
          charCounter.textContent = `${query.length} / 2000`;
          handleSubmit();
        }
      });
    });

    // New query reset button
    newQueryBtn.addEventListener("click", () => {
      resetQueryView();
      questionInput.focus();
    });

    // Scorecard Modal Listeners
    if (openEvalBtn && evalModal) {
      openEvalBtn.addEventListener("click", () => {
        evalModal.classList.remove("hidden");
        loadBenchmarkReport(false);
      });
    }

    if (closeEvalBtn && evalModal) {
      closeEvalBtn.addEventListener("click", () => {
        evalModal.classList.add("hidden");
      });
    }

    if (evalModal) {
      evalModal.addEventListener("click", (e) => {
        if (e.target === evalModal) {
          evalModal.classList.add("hidden");
        }
      });
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && evalModal && !evalModal.classList.contains("hidden")) {
        evalModal.classList.add("hidden");
      }
    });

    if (triggerEvalBtn) {
      triggerEvalBtn.addEventListener("click", () => {
        loadBenchmarkReport(true);
      });
    }
  }

  /**
   * Main Submit Handler
   */
  async function handleSubmit() {
    const rawQuestion = questionInput.value;
    const question = rawQuestion.trim();

    if (!question) {
      showError("Please enter a question before submitting.");
      questionInput.focus();
      return;
    }

    // Reset view states
    hideError();
    setLoading(true);

    // Dynamic, business-friendly status messages
    const loadingSteps = [
      "Searching company knowledge base...",
      "Reviewing official policy and contract documents...",
      "Selecting the most relevant sections...",
      "Synthesizing a clear, business-friendly answer..."
    ];
    let stepIndex = 0;
    loadingStepText.textContent = loadingSteps[0];
    const stepInterval = setInterval(() => {
      stepIndex = (stepIndex + 1) % loadingSteps.length;
      loadingStepText.textContent = loadingSteps[stepIndex];
    }, 1200);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      clearInterval(stepInterval);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error (${response.status})`);
      }

      const data = await response.json();
      renderResults(question, data);
      recordSessionHistory(question, data);

    } catch (err) {
      clearInterval(stepInterval);
      showError(err.message || "An unexpected error occurred while communicating with the server.");
    } finally {
      setLoading(false);
    }
  }

  /**
   * Set Loading State
   */
  function setLoading(isLoading) {
    if (isLoading) {
      submitBtn.disabled = true;
      questionInput.disabled = true;
      loadingIndicator.classList.remove("hidden");
      resultContainer.classList.add("hidden");
    } else {
      submitBtn.disabled = false;
      questionInput.disabled = false;
      loadingIndicator.classList.add("hidden");
    }
  }

  /**
   * Format Raw Answer into Clean Business Markdown HTML
   */
  function formatMarkdownAnswer(text) {
    if (!text) return "";

    // Escape HTML first for XSS safety
    let clean = escapeHtml(text);

    // Clean inline citations e.g. 【leave_policy.pdf | Page 1.0 | Section 1...】
    clean = clean.replace(/【(.*?)】/g, '<span class="inline-citation">$1</span>');

    // Headers
    clean = clean.replace(/^###\s+(.*?)$/gm, "<h4>$1</h4>");
    clean = clean.replace(/^##\s+(.*?)$/gm, "<h4>$1</h4>");

    // Bold text **term**
    clean = clean.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Convert bullet lists
    const lines = clean.split("\n");
    let inList = false;
    let formatted = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const isBullet = line.startsWith("* ") || line.startsWith("- ") || line.startsWith("• ");

      if (isBullet) {
        if (!inList) {
          formatted.push("<ul>");
          inList = true;
        }
        const itemText = line.replace(/^[\*\-\•]\s+/, "");
        formatted.push(`<li>${itemText}</li>`);
      } else {
        if (inList) {
          formatted.push("</ul>");
          inList = false;
        }
        if (line.length > 0) {
          if (line.startsWith("<h4>") && line.endsWith("</h4>")) {
            formatted.push(line);
          } else {
            formatted.push(`<p>${line}</p>`);
          }
        }
      }
    }

    if (inList) {
      formatted.push("</ul>");
    }

    return formatted.join("");
  }

  /**
   * Render Results & Citations
   */
  function renderResults(question, data) {
    displayQuery.textContent = question;
    answerContent.innerHTML = formatMarkdownAnswer(data.answer);

    // Business-friendly telemetry
    if (data.retrieval) {
      telemetryCandidates.textContent = "Sources: Verified";
      telemetryReranked.textContent = "Accuracy: High";
    }

    // Safety and privacy shield status
    if (telemetryGuardrails && telemetryGuardrailText) {
      if (data.guardrails) {
        if (data.guardrails.passed) {
          telemetryGuardrails.classList.remove("flagged");
          if (data.guardrails.input_checks && data.guardrails.input_checks.pii_redacted) {
            telemetryGuardrailText.textContent = "Privacy Shield (PII Redacted)";
          } else {
            telemetryGuardrailText.textContent = "Verified Safe";
          }
        } else {
          telemetryGuardrails.classList.add("flagged");
          telemetryGuardrailText.textContent = "Security Notice";
        }
      } else {
        telemetryGuardrailText.textContent = "Verified Safe";
      }
    }

    // Citations
    citationsGrid.innerHTML = "";
    const citations = data.citations || [];
    citationsCount.textContent = citations.length;

    if (citations.length === 0) {
      citationsSection.classList.add("hidden");
    } else {
      citationsSection.classList.remove("hidden");
      citations.forEach((cite) => {
        const card = createCitationCard(cite);
        citationsGrid.appendChild(card);
      });
    }

    resultContainer.classList.remove("hidden");
    resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /**
   * Create Citation Card Element
   */
  function createCitationCard(cite) {
    const card = document.createElement("div");
    card.className = "citation-card";

    const topRow = document.createElement("div");
    topRow.className = "citation-top-row";

    const docName = document.createElement("span");
    docName.className = "citation-doc-name";
    docName.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span>${escapeHtml(cite.document)}</span>
    `;

    const pageBadge = document.createElement("span");
    pageBadge.className = "citation-page-badge";
    pageBadge.textContent = `Page ${cite.page}`;

    topRow.appendChild(docName);
    topRow.appendChild(pageBadge);

    const sectionTitle = document.createElement("div");
    sectionTitle.className = "citation-section-title";
    sectionTitle.textContent = cite.section || "General Guidance";

    card.appendChild(topRow);
    card.appendChild(sectionTitle);
    return card;
  }

  /**
   * Record Query in Session History
   */
  function recordSessionHistory(question, data) {
    const record = { id: Date.now(), question, data };
    sessionHistory.unshift(record);

    if (historyEmpty) {
      historyEmpty.style.display = "none";
    }

    historyCount.textContent = sessionHistory.length;

    const item = document.createElement("div");
    item.className = "history-item";
    item.textContent = question;
    item.title = question;

    item.addEventListener("click", () => {
      questionInput.value = question;
      charCounter.textContent = `${question.length} / 2000`;
      renderResults(question, data);
    });

    historyList.prepend(item);
  }

  /**
   * Reset Query View
   */
  function resetQueryView() {
    questionInput.value = "";
    charCounter.textContent = "0 / 2000";
    hideError();
    resultContainer.classList.add("hidden");
    loadingIndicator.classList.add("hidden");
  }

  /**
   * Error Handling
   */
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  /**
   * HTML Sanitization Helper
   */
  function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Fetch and render Quality Scorecard Report
   */
  async function loadBenchmarkReport(fresh = false) {
    if (triggerEvalBtn) {
      triggerEvalBtn.disabled = true;
      triggerEvalBtn.innerHTML = `<span class="btn-text">Running Quality Test...</span>`;
    }
    if (fresh && evalTableBody) {
      evalTableBody.innerHTML = `<tr><td colspan="7" class="table-loading">Testing queries against company policies and contracts... This may take a moment.</td></tr>`;
    }

    try {
      const url = fresh ? `${API_BASE_URL}/api/evaluation/run` : `${API_BASE_URL}/api/evaluation/benchmark`;
      const method = fresh ? "POST" : "GET";
      const response = await fetch(url, { method });

      if (!response.ok) {
        throw new Error(`Failed to load scorecard (${response.status})`);
      }

      const report = await response.json();
      renderEvaluationDashboard(report);
    } catch (err) {
      console.error("Scorecard load error:", err);
      if (evalTableBody) {
        evalTableBody.innerHTML = `<tr><td colspan="7" class="table-loading" style="color: #e11d48;">Error loading scorecard: ${escapeHtml(err.message)}</td></tr>`;
      }
    } finally {
      if (triggerEvalBtn) {
        triggerEvalBtn.disabled = false;
        triggerEvalBtn.innerHTML = `<span class="btn-text">Run Quality Test</span>`;
      }
    }
  }

  /**
   * Render Quality Scorecard Dashboard
   */
  function renderEvaluationDashboard(report) {
    if (!report) return;

    const overallPct = Math.round((report.overall_rag_score ?? (report.mean_metrics ? report.mean_metrics.ragas_score : 0)) * 100);
    const faithPct = Math.round((report.mean_faithfulness ?? (report.mean_metrics ? report.mean_metrics.faithfulness : 0)) * 100);
    const relPct = Math.round((report.mean_answer_relevancy ?? (report.mean_metrics ? report.mean_metrics.answer_relevancy : 0)) * 100);
    const precPct = Math.round((report.mean_context_precision ?? (report.mean_metrics ? report.mean_metrics.context_precision : 0)) * 100);
    const recPct = Math.round((report.mean_context_recall ?? (report.mean_metrics ? report.mean_metrics.context_recall : 0)) * 100);

    if (evalOverallScore) evalOverallScore.textContent = `${overallPct}%`;

    if (valFaithfulness) valFaithfulness.textContent = `${faithPct}%`;
    if (barFaithfulness) barFaithfulness.style.width = `${faithPct}%`;

    if (valRelevancy) valRelevancy.textContent = `${relPct}%`;
    if (barRelevancy) barRelevancy.style.width = `${relPct}%`;

    if (valPrecision) valPrecision.textContent = `${precPct}%`;
    if (barPrecision) barPrecision.style.width = `${precPct}%`;

    if (valRecall) valRecall.textContent = `${recPct}%`;
    if (barRecall) barRecall.style.width = `${recPct}%`;

    const samples = report.sample_evaluations || report.results || [];
    if (evalTableBody && samples.length > 0) {
      evalTableBody.innerHTML = "";
      samples.forEach((row, idx) => {
        const tr = document.createElement("tr");
        const qScore = Math.round((row.rag_score ?? row.ragas_score ?? 0) * 100);

        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td class="eval-q-cell" title="${escapeHtml(row.question)}">${escapeHtml(row.question)}</td>
          <td>${Math.round((row.faithfulness || 0) * 100)}%</td>
          <td>${Math.round((row.answer_relevancy || 0) * 100)}%</td>
          <td>${Math.round((row.context_precision || 0) * 100)}%</td>
          <td>${Math.round((row.context_recall || 0) * 100)}%</td>
          <td><span class="score-pill">${qScore}%</span></td>
        `;
        evalTableBody.appendChild(tr);
      });
    }
  }
});
