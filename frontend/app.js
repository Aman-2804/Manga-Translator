const API = window.location.origin;

const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const fileList       = document.getElementById("file-list");
const translateBtn   = document.getElementById("translate-btn");
const uploadSection  = document.getElementById("upload-section");
const progressSection = document.getElementById("progress-section");
const progressLabel  = document.getElementById("progress-label");
const progressPct    = document.getElementById("progress-pct");
const progressFill   = document.getElementById("progress-fill");
const resultsSection = document.getElementById("results-section");
const downloadAllBtn = document.getElementById("download-all-btn");
const newTransBtn    = document.getElementById("new-translation-btn");
const retranslateBtn = document.getElementById("retranslate-btn");
const toastContainer = document.getElementById("toast-container");
const scrollLeft     = document.getElementById("compare-scroll-left");
const scrollRight    = document.getElementById("compare-scroll-right");
const targetLang     = document.getElementById("target-lang");
const dropZoneEmpty   = document.getElementById("drop-zone-empty");
const dropZonePreview = document.getElementById("drop-zone-preview");
const dropZoneTitle   = document.getElementById("drop-zone-title");
const dropZoneImages  = document.getElementById("drop-zone-images");
const progressPreview = document.getElementById("progress-preview");

let selectedFiles = [];
let originalUrls  = [];
let jobId         = null;
let pollTimer     = null;

/* ── Toast ────────────────────────────────────────── */

function toast(message, type = "error") {
  const el = document.createElement("div");
  el.className = `toast toast--${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

/* ── Drag & Drop ──────────────────────────────────── */

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  handleFiles(e.dataTransfer.files);
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
});

function isPDF(file) {
  if (file.type === "application/pdf") return true;
  if (file.name.toLowerCase().endsWith(".pdf")) return true;
  // Files with no extension and no image MIME type are treated as PDF
  const hasImageType = file.type.startsWith("image/");
  const hasKnownExt = /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name);
  if (!hasImageType && !hasKnownExt) return true;
  return false;
}

function handleFiles(files) {
  selectedFiles = Array.from(files);
  originalUrls.filter(Boolean).forEach((u) => URL.revokeObjectURL(u));

  if (selectedFiles.length === 0) return;

  const hasPDF = selectedFiles.some(isPDF);

  // Build originalUrls: null for PDFs (no client-side preview), blob URL for images
  originalUrls = selectedFiles.map((f) => isPDF(f) ? null : URL.createObjectURL(f));

  dropZoneEmpty.hidden = true;
  dropZonePreview.hidden = false;
  dropZoneImages.innerHTML = "";

  if (hasPDF) {
    // PDF: show a document icon + filename instead of image thumbnails
    const pdfFiles = selectedFiles.filter(isPDF);
    dropZoneTitle.textContent = `${pdfFiles.map(f => f.name).join(", ")}`;
    pdfFiles.forEach((f) => {
      const div = document.createElement("div");
      div.className = "pdf-preview-card";
      div.innerHTML = `<span class="pdf-preview-card__icon">&#128196;</span><span class="pdf-preview-card__name">${f.name}</span>`;
      dropZoneImages.appendChild(div);
    });
  } else {
    dropZoneTitle.textContent = `${selectedFiles.length} page(s) selected`;
    originalUrls.forEach((url) => {
      if (!url) return;
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Preview";
      dropZoneImages.appendChild(img);
    });
  }

  fileList.innerHTML = "";
  selectedFiles.forEach((f) => {
    const tag = document.createElement("span");
    tag.className = "file-tag";
    tag.textContent = f.name;
    fileList.appendChild(tag);
  });
  fileList.hidden = false;
  translateBtn.disabled = false;
}

/* ── Upload + Translate ───────────────────────────── */

translateBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;

  translateBtn.disabled = true;
  translateBtn.textContent = "Uploading…";
  resultsSection.hidden = true;
  uploadSection.hidden = true;
  progressSection.hidden = true;

  const form = new FormData();
  selectedFiles.forEach((f) => form.append("files", f));

  try {
    const upRes = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (!upRes.ok) {
      const err = await upRes.json();
      throw new Error(err.detail || "Upload failed");
    }
    const upData = await upRes.json();
    jobId = upData.job_id;

    const target = targetLang.value;
    const trRes = await fetch(
      `${API}/translate/${jobId}?target_lang=${target}&source_lang=ja`,
      { method: "POST" }
    );
    if (!trRes.ok) {
      const err = await trRes.json();
      throw new Error(err.detail || "Translation start failed");
    }

    translateBtn.textContent = "Translate";
    progressSection.hidden = false;
    progressPreview.innerHTML = "";
    originalUrls.forEach((url) => {
      if (!url) return;
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Translating…";
      progressPreview.appendChild(img);
    });
    setProgress(0, "Starting translation…");
    startPolling();
  } catch (err) {
    toast(err.message);
    translateBtn.disabled = false;
    translateBtn.textContent = "Translate";
    uploadSection.hidden = false;
    progressSection.hidden = true;
  }
});

/* ── Polling ──────────────────────────────────────── */

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 1000);
}

async function pollStatus() {
  if (!jobId) return;

  try {
    const res = await fetch(`${API}/status/${jobId}`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.status === "processing") {
      const pct = data.progress || 0;
      const label = data.step_label || "Processing…";
      setProgress(pct, label);
    }

    if (data.status === "done") {
      clearInterval(pollTimer);
      setProgress(100, "Done!");
      uploadSection.hidden = true;
      progressSection.hidden = true;
      showResults(data);
    }

    if (data.status === "failed") {
      clearInterval(pollTimer);
      toast(data.error || "Translation failed");
      uploadSection.hidden = false;
      progressSection.hidden = true;
      translateBtn.disabled = false;
    }
  } catch (err) {
    /* network hiccup — keep polling */
  }
}

function setProgress(pct, label) {
  progressFill.style.width = `${pct}%`;
  progressPct.textContent = `${pct}%`;
  if (label) progressLabel.textContent = label;
}

/* ── Results ──────────────────────────────────────── */

function showResults(data) {
  resultsSection.hidden = false;

  populateCompareView(data);

  if (data.pdf) {
    downloadAllBtn.onclick = () => {
      window.open(`${API}${data.pdf}`, "_blank");
    };
  }

  translateBtn.disabled = false;
}

/* ── Compare View (sync-scroll) ──────────────────── */

let syncScrollActive = false;

function populateCompareView(data) {
  scrollLeft.innerHTML = "";
  scrollRight.innerHTML = "";

  const hasPDFUpload = originalUrls.every((u) => u === null);

  data.pages.forEach((pageUrl, i) => {
    const originalSrc = originalUrls[i] || null;
    const translatedSrc = `${API}${pageUrl}`;

    if (originalSrc) {
      const img = document.createElement("img");
      img.src = originalSrc;
      img.alt = `Original page ${i + 1}`;
      img.loading = "lazy";
      scrollLeft.appendChild(img);
    } else {
      // PDF upload — no client-side original; show a placeholder
      const ph = document.createElement("div");
      ph.className = "pdf-original-placeholder";
      ph.textContent = `Page ${i + 1}`;
      scrollLeft.appendChild(ph);
    }

    const img2 = document.createElement("img");
    img2.src = translatedSrc;
    img2.alt = `Translated page ${i + 1}`;
    img2.loading = "lazy";
    scrollRight.appendChild(img2);
  });

  setupSyncScroll();
}

function setupSyncScroll() {
  if (syncScrollActive) return;
  syncScrollActive = true;

  let ticking = false;
  let source = null;

  function syncFrom(origin, target) {
    return () => {
      if (source && source !== origin) return;
      source = origin;

      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => {
          const maxScroll = origin.scrollHeight - origin.clientHeight;
          const ratio = maxScroll > 0 ? origin.scrollTop / maxScroll : 0;
          const targetMax = target.scrollHeight - target.clientHeight;
          target.scrollTop = ratio * targetMax;

          ticking = false;
          source = null;
        });
      }
    };
  }

  scrollLeft.addEventListener("scroll", syncFrom(scrollLeft, scrollRight), { passive: true });
  scrollRight.addEventListener("scroll", syncFrom(scrollRight, scrollLeft), { passive: true });
}

/* ── Re-translate (same images, different language) ── */

retranslateBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) {
    toast("No images to re-translate. Upload new pages first.");
    return;
  }

  retranslateBtn.disabled = true;
  retranslateBtn.textContent = "Uploading…";
  resultsSection.hidden = true;
  progressSection.hidden = false;
  setProgress(0, "Re-uploading pages…");

  const form = new FormData();
  selectedFiles.forEach((f) => form.append("files", f));

  try {
    const upRes = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (!upRes.ok) {
      const err = await upRes.json();
      throw new Error(err.detail || "Upload failed");
    }
    const upData = await upRes.json();
    jobId = upData.job_id;

    const target = targetLang.value;
    const trRes = await fetch(
      `${API}/translate/${jobId}?target_lang=${target}&source_lang=ja`,
      { method: "POST" }
    );
    if (!trRes.ok) {
      const err = await trRes.json();
      throw new Error(err.detail || "Translation start failed");
    }

    progressPreview.innerHTML = "";
    originalUrls.forEach((url) => {
      if (!url) return;
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Translating…";
      progressPreview.appendChild(img);
    });
    setProgress(0, "Starting translation…");
    syncScrollActive = false;
    startPolling();
  } catch (err) {
    toast(err.message);
    resultsSection.hidden = false;
    progressSection.hidden = true;
  } finally {
    retranslateBtn.disabled = false;
    retranslateBtn.textContent = "Translate";
  }
});

/* ── New Translation ─────────────────────────────── */

newTransBtn.addEventListener("click", () => {
  resultsSection.hidden = true;
  uploadSection.hidden = false;

  selectedFiles = [];
  originalUrls.filter(Boolean).forEach((u) => URL.revokeObjectURL(u));
  originalUrls = [];
  jobId = null;
  scrollLeft.innerHTML = "";
  scrollRight.innerHTML = "";
  syncScrollActive = false;
  fileList.innerHTML = "";
  fileList.hidden = true;
  fileInput.value = "";
  translateBtn.disabled = true;
  translateBtn.textContent = "Translate";
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  progressLabel.textContent = "Preparing…";

  dropZoneEmpty.hidden = false;
  dropZonePreview.hidden = true;
  dropZoneImages.innerHTML = "";
});
