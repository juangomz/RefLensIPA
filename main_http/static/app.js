// ═══════════════════════════════════════════════════════════
//  RefLens — Aura Graph Engine  |  app.js
//  Companion script for the legacy HTML template.
//  Keeps parity with index.html logic.
// ═══════════════════════════════════════════════════════════

const $ = (id) => document.getElementById(id);
const lucide_refresh = () => lucide.createIcons();

let lastData  = null;
let activeTab = "final";
let cy        = null;
let lastQuery = "";

// ── STOPWORDS (ES + EN) ─────────────────────────────────────
const STOPWORDS = new Set([
    "que","de","la","el","en","un","una","los","las","del","con","por","para","como","más",
    "pero","sus","les","sin","sobre","entre","cuando","todo","también","me","mi","muy","ya",
    "sea","hay","fue","ser","son","está","han","esto","eso","esta","ese","esa","tiene",
    "the","and","for","are","but","not","you","all","can","had","her","was","one","our",
    "out","day","get","has","him","his","how","its","may","new","now","old","see","two",
    "way","who","did","any","been","from","this","that","with","they","have","will","what",
    "when","your","said","each","which","their","there","were","them","these","those",
    "would","could","should","about","after","being","every","other","some","into","than",
    "then","well","also","just","only","more","most","such","even","both","much","many",
    "first","second","third","last","next","long","little","own","right","high","come","went"
]);

// ── TOPIC EXTRACTION ────────────────────────────────────────
/**
 * Extract meaningful topics from chunk texts using frequency + IDF scoring.
 * Returns string[] sorted by relevance, max 12 terms.
 * Falls back to [] (never uses final_answer regex).
 */
function extractTopicsFromChunks(chunks) {
    if (!chunks || !chunks.length) return [];

    const allTexts  = chunks.map(c => (c.text || "").toLowerCase());
    const totalDocs = allTexts.length;
    const termDocFreq = {};

    allTexts.forEach(text => {
        const words    = text.match(/[a-záéíóúñ]{4,}/g) || [];
        const docTerms = new Set(words);
        docTerms.forEach(t => {
            if (!STOPWORDS.has(t)) termDocFreq[t] = (termDocFreq[t] || 0) + 1;
        });
    });

    const minFreq = Math.max(1, Math.floor(totalDocs * 0.2));
    const scored  = Object.entries(termDocFreq)
        .filter(([t, f]) => f >= minFreq && t.length >= 4)
        .map(([term, freq]) => {
            const idf      = Math.log((totalDocs + 1) / (freq + 1));
            const isProper = chunks.some(c =>
                new RegExp(`\\b${term[0].toUpperCase() + term.slice(1)}\\b`).test(c.text || "")
            );
            return {
                term:  isProper ? term[0].toUpperCase() + term.slice(1) : term,
                score: freq * (1 + idf) + (isProper ? 2 : 0)
            };
        })
        .sort((a, b) => b.score - a.score);

    // Remove substrings of higher-ranked terms
    const final = [];
    for (const item of scored) {
        const dominated = final.some(f =>
            f.term.toLowerCase().includes(item.term.toLowerCase()) ||
            item.term.toLowerCase().includes(f.term.toLowerCase())
        );
        if (!dominated) final.push(item);
        if (final.length >= 12) break;
    }
    return final.map(i => i.term);
}

// ── NODE LABEL ──────────────────────────────────────────────
function getNodeLabel(chunk) {
    if (chunk.title && chunk.title.trim() &&
        !["neo4j graph","nodo","chunk"].includes(chunk.title.trim().toLowerCase())) {
        return chunk.title.trim().slice(0, 24);
    }
    if (chunk.name) return String(chunk.name).trim().slice(0, 24);
    if (chunk.text) {
        const patterns = [
            /e\.name[:\s=]+([^\n|,;\]]+)/i,
            /canonical_name[:\s=]+([^\n|,;\]]+)/i,
            /"name"\s*:\s*"([^"]+)"/i,
            /name[:\s=]+([A-ZÁÉÍÓÚÑ][^\n|,;]{2,24})/,
        ];
        for (const re of patterns) {
            const m = chunk.text.match(re);
            if (m && m[1].trim().length > 1) return m[1].trim().slice(0, 24);
        }
        const clean = chunk.text
            .replace(/props\s*:\s*\[.*?\]/gi, "")
            .replace(/rel_type\s*:|rel_name\s*:/gi, "")
            .replace(/[^\wáéíóúñÁÉÍÓÚÑ\s]/g, " ")
            .trim();
        const words = clean.split(/\s+/).filter(w => w.length > 2);
        if (words.length >= 2) return words.slice(0, 3).join(" ").slice(0, 24);
    }
    return `#${chunk.id ?? "?"}`;
}

function getNodeColor(chunk) {
    return (chunk.source || "").toLowerCase().includes("neo4j") ? "#10b981" : "#818cf8";
}
function getNodeSize(score) {
    return score != null ? 10 + Math.round(score * 16) : 14;
}

// ── STATUS ──────────────────────────────────────────────────
function setStatus(state, message) {
    const dot  = $("status-dot");
    const text = $("status-text");
    if (!dot || !text) return;
    text.textContent = message;
    dot.className    = "h-1.5 w-1.5 rounded-full ";
    if (state === "loading") dot.classList.add("bg-indigo-500", "agent-pulse");
    else if (state === "error") dot.classList.add("bg-red-500");
    else dot.classList.add("bg-emerald-500");
}

// ── VERDICT BADGE ───────────────────────────────────────────
function renderBadge(verdict) {
    const container = $("badge-container");
    if (!container) return;
    if (!verdict)  { container.innerHTML = ""; return; }

    const styles = {
        pass:    "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
        revise:  "border-amber-500/30  bg-amber-500/10  text-amber-400",
        fail:    "border-rose-500/30   bg-rose-500/10   text-rose-400"
    };
    container.innerHTML = `
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold ${styles[verdict] || styles.fail}">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
            </span>
            AGENTE EVAL: ${verdict.toUpperCase()}
        </span>
    `;
}

// ── ESCAPE HTML ─────────────────────────────────────────────
function escapeHtml(s) {
    if (!s) return "";
    return String(s).replace(/[&<>"']/g, m =>
        ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":'&#39;' }[m])
    );
}

// ── RENDER ──────────────────────────────────────────────────
function render() {
    const out = $("out");
    if (!lastData) {
        if (out) out.innerHTML = `<div class="py-20 text-center text-zinc-600 italic">No hay datos disponibles</div>`;
        if ($("chunks")) $("chunks").innerHTML = "";
        if ($("meta"))   $("meta").textContent = "";
        renderBadge(null);
        return;
    }

    renderBadge(lastData.verdict);

    const meta = $("meta");
    if (meta) meta.textContent = `LATENCIA: ${lastData.latency_ms || "—"}ms | NODOS: ${lastData.chunks?.length || 0}`;

    // Content tabs
    if (activeTab === "final") {
        out.innerHTML = marked.parse(lastData.final_answer || "Sin respuesta.");
    } else {
        out.innerHTML = `<pre class="p-4 bg-black rounded-lg text-indigo-300 text-xs overflow-x-auto">${JSON.stringify(lastData.eval_json, null, 2)}</pre>`;
    }

    // Chunks
    const chunks = lastData.chunks || [];
    if ($("chunks")) {
        $("chunks").innerHTML = chunks.map((c, i) => {
            const isNeo  = (c.source || "").toLowerCase().includes("neo4j");
            const score  = typeof c.score === "number" ? c.score : null;
            const label  = getNodeLabel(c);
            const srcBadge = isNeo
                ? `<span style="color:#10b981;font-size:9px;font-weight:700">Neo4j</span>`
                : `<span style="color:#818cf8;font-size:9px;font-weight:700">Chroma</span>`;
            return `
            <div class="group relative p-4 rounded-xl bg-zinc-900/40 border ${isNeo ? "border-l-2 border-l-emerald-500/60" : "border-l-2 border-l-indigo-500/60"} border-zinc-800/50 hover:border-indigo-500/50 transition-all duration-300 cursor-pointer"
                 onclick="focusNode(${i})">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded">#${c.id ? String(c.id).slice(0,8) : i}</span>
                        ${srcBadge}
                    </div>
                    <span class="text-[10px] text-zinc-500 font-bold tracking-tighter">${score ? (score * 100).toFixed(1) + "%" : ""}</span>
                </div>
                <h4 class="text-xs font-semibold text-zinc-200 mb-2 truncate" title="${escapeHtml(label)}">${escapeHtml(label)}</h4>
                <p class="text-[11px] text-zinc-500 leading-relaxed line-clamp-4 group-hover:line-clamp-none transition-all">${escapeHtml(c.text || "Sin contenido")}</p>
                ${score != null ? `<div class="w-full h-0.5 bg-zinc-800 rounded-full mt-3 overflow-hidden"><div class="h-full ${isNeo ? "bg-emerald-500" : "bg-indigo-500"} rounded-full" style="width:${(score*100).toFixed(0)}%"></div></div>` : ""}
            </div>`;
        }).join("");
    }

    // Topics
    renderTopics(chunks, lastData.topics);

    // Graph title
    updateGraphTitle();

    // Graph
    initGraph(chunks);
    lucide_refresh();
}

// ── TOPICS PANEL ────────────────────────────────────────────
function renderTopics(chunks, backendTopics) {
    const panel = $("topics-panel");
    if (!panel) return;

    let topics      = [];
    let originLabel = "";

    if (backendTopics && backendTopics.length) {
        topics      = backendTopics.slice(0, 12);
        originLabel = "backend";
    } else if (chunks.length) {
        topics      = extractTopicsFromChunks(chunks);
        originLabel = "extraído";
    }

    if (!topics.length) { panel.classList.add("hidden"); return; }

    const badge  = $("topics-origin");
    const tagsEl = $("topics-tags");
    if (badge)  badge.textContent  = originLabel;
    if (tagsEl) tagsEl.innerHTML   = topics.map(t =>
        `<span class="inline-flex items-center px-2 py-0.5 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[10px] font-mono font-semibold">${escapeHtml(t)}</span>`
    ).join("");
    panel.classList.remove("hidden");
}

// ── GRAPH TITLE ─────────────────────────────────────────────
function updateGraphTitle() {
    const titleEl = $("graph-title");
    if (!titleEl) return;
    if (lastQuery) {
        titleEl.textContent = `Grafo de Conocimiento · "${lastQuery.length > 35 ? lastQuery.slice(0,35)+"…" : lastQuery}"`;
    } else {
        titleEl.textContent = "Grafo de Conocimiento";
    }
}

// ── GRAPH ───────────────────────────────────────────────────
function initGraph(chunks) {
    const container = $("cy");
    if (!container) return;

    try {
        const valid = (chunks || []).filter(c => c.text && c.text !== "Sin contenido");
        const nodes = valid.map((c, i) => ({
            data: {
                id:       `n${i}`,
                label:    getNodeLabel(c),
                fullText: (c.text || "").slice(0, 160),
                score:    c.score,
                source:   c.source || "chroma",
                color:    getNodeColor(c),
                size:     getNodeSize(c.score),
                idx:      i
            }
        }));

        const edges = [];
        for (let i = 0; i < nodes.length - 1; i++)
            edges.push({ data: { source: `n${i}`, target: `n${i+1}` } });
        // Extra edges for same-source, high-score pairs
        for (let i = 0; i < valid.length; i++) {
            for (let j = i + 2; j < valid.length; j++) {
                if (valid[i].source === valid[j].source &&
                    (valid[i].score||0) > 0.72 && (valid[j].score||0) > 0.72 &&
                    edges.length < nodes.length * 2.5)
                    edges.push({ data: { source: `n${i}`, target: `n${j}` } });
            }
        }

        if (cy) cy.destroy();
        cy = cytoscape({
            container,
            elements: { nodes, edges },
            style: [
                {
                    selector: "node",
                    style: {
                        "background-color": "data(color)",
                        "label":            "data(label)",
                        "color":            "#94a3b8",
                        "font-size":        "9px",
                        "font-family":      "'JetBrains Mono', monospace",
                        "width":            n => (n.data("size") || 14),
                        "height":           n => (n.data("size") || 14),
                        "text-margin-y":    8,
                        "text-valign":      "bottom",
                        "border-width":     0,
                        "transition-property": "opacity, width, height",
                        "transition-duration": "0.15s"
                    }
                },
                { selector: "node.highlighted", style: { "border-width": 2, "border-color": "#f8fafc", "color": "#e2e8f0", "z-index": 10 } },
                { selector: "node.dimmed",       style: { "opacity": 0.15 } },
                {
                    selector: "edge",
                    style: {
                        "width":               1,
                        "line-color":          "#334155",
                        "target-arrow-shape":  "none",
                        "curve-style":         "bezier",
                        "opacity":             0.45
                    }
                },
                { selector: "edge.highlighted", style: { "line-color": "#475569", "opacity": 0.9 } },
                { selector: "edge.dimmed",       style: { "opacity": 0.05 } }
            ],
            layout: { name: "cose", padding: 30, animate: true, animationDuration: 450 }
        });

        // Node interactions
        cy.on("tap", "node", evt => {
            const t = evt.target;
            cy.elements().removeClass("highlighted dimmed");
            const hood = t.closedNeighborhood();
            cy.elements().not(hood).addClass("dimmed");
            hood.addClass("highlighted");
            highlightChunk(t.data("idx") ?? -1);
        });
        cy.on("tap", evt => {
            if (evt.target === cy) {
                cy.elements().removeClass("highlighted dimmed");
                highlightChunk(-1);
            }
        });

        // Stats
        const scores = valid.map(c => c.score).filter(s => s != null);
        if ($("gs-nodes")) $("gs-nodes").textContent = nodes.length;
        if ($("gs-edges")) $("gs-edges").textContent = edges.length;
        if ($("gs-avg"))   $("gs-avg").textContent   = scores.length
            ? (scores.reduce((a,b)=>a+b,0)/scores.length*100).toFixed(0)+"%"
            : "—";

    } catch (e) {
        console.error("Error en Grafo:", e);
    }
}

// ── CHUNK HIGHLIGHT ─────────────────────────────────────────
function highlightChunk(idx) {
    document.querySelectorAll(".chunk-card, #chunks > div").forEach((el, i) => {
        if (idx === -1) { el.style.opacity = "1"; }
        else if (i === idx) { el.style.opacity = "1"; el.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
        else { el.style.opacity = "0.2"; }
    });
}

window.focusNode  = idx => {
    if (!cy) return;
    const n = cy.$(`#n${idx}`);
    if (!n.length) return;
    cy.elements().removeClass("highlighted dimmed");
    const hood = n.closedNeighborhood();
    cy.elements().not(hood).addClass("dimmed");
    hood.addClass("highlighted");
    highlightChunk(idx);
};
window.resetZoom = () => { if (cy) cy.fit(); };

// ── API ──────────────────────────────────────────────────────
async function handleAsk() {
    const query = $("query").value.trim();
    if (!query) return;
    lastQuery = query;

    setStatus("loading", "Agentes razonando...");
    $("ask").disabled = true;

    try {
        const response = await fetch("/api/ask", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ query, top_k: 10, show_debug: true })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        lastData = await response.json();
        render();
        setStatus("ready", "Análisis completado");
    } catch (e) {
        setStatus("error", e.message);
    } finally {
        $("ask").disabled = false;
    }
}

// ── EVENTS ──────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll(".tab-btn").forEach(b =>
            b.classList.remove("text-indigo-400", "bg-indigo-500/10")
        );
        btn.classList.add("text-indigo-400", "bg-indigo-500/10");
        activeTab = btn.dataset.tab;
        render();
    };
});

if ($("ask"))   $("ask").onclick   = handleAsk;
if ($("clear")) $("clear").onclick = () => {
    lastData  = null;
    lastQuery = "";
    render();
    setStatus("ready", "Sesión limpia");
    if ($("query")) $("query").value = "";
    if (cy) { cy.destroy(); cy = null; }
    updateGraphTitle();
};

if ($("query")) {
    $("query").onkeydown = e => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleAsk(); };
}