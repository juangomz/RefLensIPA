const $ = (id) => document.getElementById(id);
const lucide_refresh = () => lucide.createIcons();

let lastData = null;
let activeTab = "final";

// Actualizar label de top_k
$("topk").oninput = (e) => $("topk-val").textContent = e.target.value;

function setStatus(state, message) {
    const dot = $("status-dot");
    const text = $("status-text");
    text.textContent = message;
    
    dot.className = "h-1.5 w-1.5 rounded-full ";
    if (state === "loading") {
        dot.classList.add("bg-indigo-500", "agent-pulse");
    } else if (state === "error") {
        dot.classList.add("bg-red-500");
    } else {
        dot.classList.add("bg-emerald-500");
    }
}

function renderBadge(verdict) {
    const container = $("badge-container");
    if (!verdict) { container.innerHTML = ""; return; }

    const styles = {
        pass: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
        revise: "border-amber-500/30 bg-amber-500/10 text-amber-400",
        fail: "border-rose-500/30 bg-rose-500/10 text-rose-400"
    };

    container.innerHTML = `
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold ${styles[verdict] || styles.fail}">
            <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
            </span>
            AGENTE QA: ${verdict.toUpperCase()}
        </span>
    `;
}

function render() {
    const out = $("out");
    if (!lastData) {
        out.innerHTML = `<div class="py-20 text-center text-zinc-600 italic">No hay datos disponibles</div>`;
        $("chunks").innerHTML = "";
        $("meta").textContent = "";
        renderBadge(null);
        return;
    }

    renderBadge(lastData.verdict);
    $("meta").textContent = `LATENCIA: ${lastData.latency_ms}ms | NODOS: ${lastData.chunks?.length || 0}`;

    // Selección de Tab
    if (activeTab === "final") out.textContent = lastData.final_answer || "";
    else if (activeTab === "draft") out.textContent = lastData.draft_answer || "";
    else out.innerHTML = `<pre class="p-4 bg-black rounded-lg text-indigo-300 text-xs overflow-x-auto">${JSON.stringify(lastData.qa_json, null, 2)}</pre>`;

    // Render de Chunks mejorado
    $("chunks").innerHTML = (lastData.chunks || []).map((c, i) => `
        <div class="group relative p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/50 hover:border-indigo-500/50 transition-all duration-300">
            <div class="flex items-center justify-between mb-2">
                <span class="text-[10px] font-mono text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded">#${c.id || i}</span>
                <span class="text-[10px] text-zinc-500 font-bold tracking-tighter">${c.score ? (c.score * 100).toFixed(1) + '%' : ''}</span>
            </div>
            <h4 class="text-xs font-semibold text-zinc-300 mb-2 truncate">${c.title || 'Nodo de Conocimiento'}</h4>
            <p class="text-[11px] text-zinc-500 leading-relaxed line-clamp-4 group-hover:line-clamp-none transition-all">${escapeHtml(c.text)}</p>
        </div>
    `).join("");
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

async function handleAsk() {
    const query = $("query").value.trim();
    if (!query) return;

    setStatus("loading", "Agentes razonando...");
    $("ask").disabled = true;

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                query,
                top_k: parseInt($("topk").value),
                show_debug: $("debug").checked
            })
        });

        if (!response.ok) throw new Error(`Status ${response.status}`);

        lastData = await response.json();
        render();
        setStatus("ready", "Análisis completado");
    } catch (e) {
        setStatus("error", e.message);
    } finally {
        $("ask").disabled = false;
    }
}

// Event Listeners
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("text-indigo-400", "bg-indigo-500/10"));
        btn.classList.add("text-indigo-400", "bg-indigo-500/10");
        activeTab = btn.dataset.tab;
        render();
    };
});

$("ask").onclick = handleAsk;
$("clear").onclick = () => { lastData = null; render(); setStatus("ready", "Sesión limpia"); $("query").value = ""; };

// Shortcut: Cmd/Ctrl + Enter para enviar
$("query").onkeydown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleAsk();
};