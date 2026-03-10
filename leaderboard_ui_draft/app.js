const TASKS = {
  SCA_QA: {
    label: "SCA-QA",
    metricLabel: "Speech Context Faithfulness",
    shortMetric: "Faithfulness",
    lowerBetter: false,
    datasets: ["History_before_chosun", "History_after_chosun", "K-sports", "K-pop"],
  },
  SPEECH_QA: {
    label: "Speech QA",
    metricLabel: "Accuracy (%)",
    shortMetric: "Acc(%)",
    lowerBetter: false,
    datasets: ["CLICk", "KoBest BoolQ"],
  },
  SPEECH_INSTRUCTION: {
    label: "Speech Instruction",
    metricLabel: "GPT-4o Judge Score",
    shortMetric: "GPT-4o Judge",
    lowerBetter: false,
    datasets: ["KUDGE", "Vicuna", "OpenHermes", "Alpaca"],
  },
  ASR: {
    label: "ASR",
    metricLabel: "CER (%)",
    shortMetric: "CER",
    lowerBetter: true,
    datasets: ["KsponSpeech", "CommonVoice-KO", "Zeroth-Korean"],
  },
  TRANSLATION: {
    label: "Translation",
    metricLabel: "BLEU / METEOR",
    shortMetric: "BLEU/METEOR",
    lowerBetter: false,
    datasets: ["ETRI-TST-Common", "ETRI-TST-HE"],
  },
  LONG_SPEECH_UNDERSTANDING: {
    label: "Long Speech Understanding",
    metricLabel: "Accuracy (%)",
    shortMetric: "Acc(%)",
    lowerBetter: false,
    datasets: ["MCTest"],
  },
};

const TASK_KEYS = Object.keys(TASKS);

const ui = {
  taskMenu: document.getElementById("taskMenu"),
  submitToggleBtn: document.getElementById("submitToggleBtn"),
  controlPanel: document.getElementById("controlPanel"),
  fileInput: document.getElementById("jsonFile"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  uploadTaskSelect: document.getElementById("uploadTaskSelect"),
  rankNameInput: document.getElementById("rankNameInput"),
  modelInput: document.getElementById("modelInput"),
  urlInput: document.getElementById("urlInput"),
  homeView: document.getElementById("homeView"),
  taskView: document.getElementById("taskView"),
};

const state = {
  activeView: "HOME",
  entries: [],
  isSubmitPanelOpen: false,
};

const OVERALL_BADGE_IMAGES = {
  1: "./images/1st.png",
  2: "./images/2nd.png",
  3: "./images/3rd.png",
};

function setStatus(text) {
  return text;
}

function toNum(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function average(values) {
  const valid = values.filter((value) => value !== null);
  if (!valid.length) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function taskAverage(taskKey, taskScores) {
  const datasets = TASKS[taskKey].datasets;
  return average(
    datasets.map((dataset) => toNum(taskScores ? taskScores[dataset] : null)),
  );
}

function normalizeJsonArray(input) {
  if (!Array.isArray(input)) {
    throw new Error("Top-level JSON must be an array.");
  }

  return input.map((item, index) => {
    const tasks = {};
    TASK_KEYS.forEach((taskKey) => {
      const taskNode =
        (item.tasks && item.tasks[taskKey]) ||
        (item.dataset_scores && item.dataset_scores[taskKey]) ||
        {};
      const normalizedTask = {};

      if (taskNode.datasets) {
        TASKS[taskKey].datasets.forEach((dataset) => {
          normalizedTask[dataset] = toNum(taskNode.datasets[dataset]);
        });
      } else {
        TASKS[taskKey].datasets.forEach((dataset) => {
          normalizedTask[dataset] = toNum(taskNode[dataset]);
        });
      }

      tasks[taskKey] = normalizedTask;
    });

    return {
      id: item.id || `entry-${index + 1}`,
      rank_name: item.rank_name || item.rankName || `Entry ${index + 1}`,
      model: item.model || "",
      url: item.url || "",
      status: item.status || "scored",
      source: item.source || "aggregate-json",
      tasks,
    };
  });
}

function enrichEntries(entries) {
  return entries.map((entry) => {
    const taskOverall = {};
    TASK_KEYS.forEach((taskKey) => {
      taskOverall[taskKey] = taskAverage(taskKey, entry.tasks[taskKey]);
    });

    return {
      ...entry,
      taskOverall,
      overall: average(TASK_KEYS.map((taskKey) => taskOverall[taskKey])),
    };
  });
}

function metricClass(taskKey, value) {
  if (value === null) return "muted";
  return TASKS[taskKey].lowerBetter ? "metric-bad" : "metric-good";
}

function compareScores(a, b, lowerBetter) {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return lowerBetter ? a - b : b - a;
}

function sortOverall(entries) {
  return [...entries]
    .sort((a, b) => compareScores(a.overall, b.overall, false))
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

function sortTask(entries, taskKey, datasetName) {
  const lowerBetter = TASKS[taskKey].lowerBetter;
  return [...entries]
    .sort((a, b) => {
      const aValue =
        datasetName === "Overall"
          ? a.taskOverall[taskKey]
          : toNum(a.tasks[taskKey] ? a.tasks[taskKey][datasetName] : null);
      const bValue =
        datasetName === "Overall"
          ? b.taskOverall[taskKey]
          : toNum(b.tasks[taskKey] ? b.tasks[taskKey][datasetName] : null);
      return compareScores(aValue, bValue, lowerBetter);
    })
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

function entriesForTask(taskKey) {
  return state.entries.filter((entry) => {
    if (entry.status !== "pending") return true;
    return entry.pendingTask === taskKey;
  });
}

function renderMenu() {
  const items = [
    { key: "HOME", label: "Home", meta: "Overall ranking" },
    ...TASK_KEYS.map((taskKey) => ({
      key: taskKey,
      label: TASKS[taskKey].label,
      meta: `${TASKS[taskKey].datasets.length} datasets`,
    })),
  ];

  ui.taskMenu.innerHTML = items
    .map(
      (item) => `
        <button type="button" class="${state.activeView === item.key ? "active" : ""}" data-view="${item.key}">
          <span class="menu-label">${item.label}</span>
          <span class="menu-meta">${item.meta}</span>
        </button>
      `,
    )
    .join("");

  ui.taskMenu.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeView = button.dataset.view;
      renderApp();
    });
  });
}

function renderUploadTaskOptions() {
  ui.uploadTaskSelect.innerHTML = TASK_KEYS.map(
    (taskKey) => `<option value="${taskKey}">${TASKS[taskKey].label}</option>`,
  ).join("");

  if (state.activeView !== "HOME" && TASKS[state.activeView]) {
    ui.uploadTaskSelect.value = state.activeView;
  }
}

function renderControlPanel() {
  ui.controlPanel.classList.toggle("hidden", !state.isSubmitPanelOpen);
}

function renderRankStrip(entries) {
  return `
    <section class="section-card card">
      <div class="section-head">
        <h3>Top Ranking</h3>
      </div>
      <div class="rank-strip-list">
        ${entries
          .slice(0, 12)
          .map(
            (entry) => `
              <article class="rank-pill">
                <div class="rank-badge-wrap">
                  ${
                    OVERALL_BADGE_IMAGES[entry.rank]
                      ? `<img class="rank-badge-image" src="${OVERALL_BADGE_IMAGES[entry.rank]}" alt="${entry.rank} place" />`
                      : `<span class="rank-badge">#${entry.rank}</span>`
                  }
                </div>
                <span class="rank-name">${entry.rank_name}</span>
              </article>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderHomeTable(entries) {
  const topRow = [
    '<th rowspan="2" class="rank-col">Rank</th>',
    '<th rowspan="2">RankName</th>',
    '<th rowspan="2">Model</th>',
    '<th rowspan="2">URL</th>',
    '<th rowspan="2">Overall</th>',
    ...TASK_KEYS.map(
      (taskKey) =>
        `<th class="grouped" colspan="1">${TASKS[taskKey].label}</th>`,
    ),
  ].join("");

  const subRow = TASK_KEYS.map(
    (taskKey) => `<th>${TASKS[taskKey].shortMetric}</th>`,
  ).join("");

  const bodyRows = entries
    .map((entry) => {
      const urlCell = entry.url
        ? `<a class="url-link" href="${entry.url}" target="_blank" rel="noopener noreferrer" aria-label="External link"><img src="./images/external-link.png" alt="" /></a>`
        : "-";
      const taskCells = TASK_KEYS.map((taskKey) => {
        const value = entry.taskOverall[taskKey];
        return `<td><span class="${metricClass(taskKey, value)}">${value === null ? "-" : value.toFixed(3)}</span></td>`;
      }).join("");

      return `
        <tr>
          <td class="rank-col">${entry.rank}</td>
          <td>${entry.rank_name}</td>
          <td>${entry.model || "-"}</td>
          <td>${urlCell}</td>
          <td>${entry.overall === null ? "-" : entry.overall.toFixed(3)}</td>
          ${taskCells}
        </tr>
      `;
    })
    .join("");

  return `
    <section class="section-card card">
      <div class="section-head">
        <h3>Overall Leaderboard</h3>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>${topRow}</tr>
            <tr>${subRow}</tr>
          </thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderHome() {
  const ranked = sortOverall(state.entries);
  ui.homeView.innerHTML = `${renderRankStrip(ranked)}${renderHomeTable(ranked)}`;
}

function renderTask() {
  const taskKey = state.activeView;
  const task = TASKS[taskKey];
  const datasetOptions = ["Overall", ...task.datasets];
  const initialDataset = datasetOptions[0];

  ui.taskView.innerHTML = `
    <section class="section-card card">
      <div class="section-head">
        <div>
          <h3 class="task-title">Task : ${task.label}</h3>
        </div>
      </div>
      <div class="task-filters">
        <label class="field">
          <span>Dataset</span>
          <select id="taskDatasetSelect">
            ${datasetOptions
              .map((dataset) => `<option value="${dataset}">${dataset}</option>`)
              .join("")}
          </select>
        </label>
        <label class="field">
          <span>Metric</span>
          <select id="taskMetricSelect" disabled>
            <option>${task.metricLabel}</option>
          </select>
        </label>
      </div>
      <div id="taskTableMount"></div>
    </section>
  `;

  const datasetSelect = document.getElementById("taskDatasetSelect");
  const tableMount = document.getElementById("taskTableMount");

  function drawTaskTable(datasetName) {
    const rows = sortTask(entriesForTask(taskKey), taskKey, datasetName);
    const headerCells = `
      <th class="rank-col col-rank">Rank</th>
      <th class="col-rankname">RankName</th>
      <th class="col-model">Model</th>
      <th>${datasetName}</th>
    `;

    const bodyRows = rows
      .map((entry) => {
        const datasetValue =
          datasetName === "Overall"
            ? toNum(entry.taskOverall[taskKey])
            : toNum(entry.tasks[taskKey] ? entry.tasks[taskKey][datasetName] : null);
        return `
          <tr>
            <td class="rank-col col-rank">${entry.rank}</td>
            <td class="col-rankname">${entry.rank_name}</td>
            <td class="col-model">${entry.model || "-"}</td>
            <td><span class="${metricClass(taskKey, datasetValue)}">${datasetValue === null ? "-" : datasetValue.toFixed(3)}</span></td>
          </tr>
        `;
      })
      .join("");

    tableMount.innerHTML = `
      <div class="table-scroll">
        <table class="task-performance-table">
          <colgroup>
            <col class="col-rank" />
            <col class="col-rankname" />
            <col class="col-model" />
            <col />
          </colgroup>
          <thead>
            <tr>${headerCells}</tr>
          </thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    `;
  }

  datasetSelect.addEventListener("change", (event) => {
    drawTaskTable(event.target.value);
  });

  drawTaskTable(initialDataset);
}

function renderApp() {
  renderMenu();
  renderUploadTaskOptions();
  renderControlPanel();

  if (state.activeView === "HOME") {
    ui.homeView.classList.remove("hidden");
    ui.taskView.classList.add("hidden");
    ui.taskView.innerHTML = "";
    renderHome();
    return;
  }

  ui.homeView.classList.add("hidden");
  ui.homeView.innerHTML = "";
  ui.taskView.classList.remove("hidden");
  renderTask();
}

function inferTaskFromViewOrSelection() {
  if (TASKS[state.activeView]) return state.activeView;
  return ui.uploadTaskSelect.value || "ASR";
}

function pendingEntryFromJsonl(lines, fileName) {
  const taskKey = inferTaskFromViewOrSelection();
  return {
    id: `pending-${Date.now()}`,
    rank_name: ui.rankNameInput.value.trim() || fileName.replace(/\.jsonl$/i, ""),
    model: ui.modelInput.value.trim() || "Pending model",
    url: ui.urlInput.value.trim(),
    status: "pending",
    source: "raw-jsonl",
    rawLineCount: lines.length,
    tasks: TASK_KEYS.reduce((acc, key) => {
      acc[key] = {};
      TASKS[key].datasets.forEach((dataset) => {
        acc[key][dataset] = null;
      });
      return acc;
    }, {}),
    pendingTask: taskKey,
  };
}

function mergeEntries(newEntries) {
  state.entries = enrichEntries([...state.entries, ...newEntries]);
  renderApp();
}

function parseJsonl(text, fileName) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  lines.forEach((line) => {
    JSON.parse(line);
  });

  const entry = pendingEntryFromJsonl(lines, fileName);
  mergeEntries([entry]);
  setStatus(
    `jsonl uploaded: ${fileName}, ${lines.length} lines, queued for ${TASKS[entry.pendingTask].label}`,
  );
}

function parseJson(text) {
  const json = JSON.parse(text);
  const entries = normalizeJsonArray(json);
  const pendingEntries = state.entries.filter((entry) => entry.status === "pending");
  state.entries = enrichEntries([...entries, ...pendingEntries]);
  renderApp();
  setStatus(
    `json loaded: ${entries.length} scored entries, keeping ${pendingEntries.length} pending entries`,
  );
}

async function loadSample() {
  try {
    const res = await fetch("./sample-data.json");
    const json = await res.text();
    parseJson(json);
  } catch (error) {
    setStatus(`sample load failed: ${error.message}`);
  }
}

ui.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  try {
    const text = await file.text();
    if (file.name.toLowerCase().endsWith(".jsonl")) {
      parseJsonl(text, file.name);
    } else {
      parseJson(text);
    }
  } catch (error) {
    setStatus(`file parsing failed: ${error.message}`);
  } finally {
    ui.fileInput.value = "";
  }
});

ui.loadSampleBtn.addEventListener("click", loadSample);
ui.submitToggleBtn.addEventListener("click", () => {
  state.isSubmitPanelOpen = !state.isSubmitPanelOpen;
  renderControlPanel();
});

renderApp();
