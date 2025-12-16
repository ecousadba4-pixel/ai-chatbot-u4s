(() => {
  const DEFAULT_ENDPOINT = "https://u4s-ai-chatbot.fantom-api.ru/api/chat";
  const STORAGE_KEY = "u4s_history_v1";
  const SESSION_KEY = "u4s_session_v1";
  const SIZE_EVENT = "u4s-iframe-size";
  const EXTRA_PADDING = 12;
  const POST_DEBOUNCE_MS = 60;
  const BURST_PULSES = 12;
  const BURST_INTERVAL_MS = 400;
  const INITIAL_GREETING =
    "Здравствуйте! Я помогу с вопросами про проживание, ресторан «Калина Красная», баню, контакты и др.";
  const CONNECTION_ERROR_MESSAGE =
    "Не удалось связаться с сервером. Проверьте подключение и попробуйте снова.";
  const RESET_ERROR_MESSAGE =
    "Не удалось сбросить диалог. Проверьте подключение и попробуйте снова.";

  const doc = document;
  const root = doc.documentElement;
  const body = doc.body;

  const elements = {
    fab: doc.getElementById("u4s-fab"),
    chat: doc.getElementById("u4s-chat"),
    scroll: doc.getElementById("u4s-scroll"),
    form: doc.getElementById("u4s-inputbar"),
    input: doc.getElementById("u4s-q"),
    reset: doc.getElementById("u4s-reset"),
  };

  if (!elements.fab || !elements.chat || !elements.scroll || !elements.form || !elements.input || !elements.reset) {
    console.warn("U4S widget: required DOM nodes are missing");
    return;
  }

  const timeFormatter = new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" });

  let memorySessionId = null;

  function createUuid() {
    if (typeof crypto !== "undefined") {
      if (typeof crypto.randomUUID === "function") {
        try {
          return crypto.randomUUID();
        } catch (_) {
          /* noop */
        }
      }
      if (typeof crypto.getRandomValues === "function") {
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"));
        return (
          `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-` +
          `${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`
        );
      }
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
      const rand = Math.floor(Math.random() * 16);
      const value = char === "x" ? rand : (rand & 0x3) | 0x8;
      return value.toString(16);
    });
  }

  function ensureSessionId() {
    if (memorySessionId) {
      return memorySessionId;
    }
    let existing = "";
    try {
      existing = window.localStorage.getItem(SESSION_KEY) || "";
    } catch (_) {
      existing = "";
    }
    if (existing && existing.trim()) {
      memorySessionId = existing.trim();
      return memorySessionId;
    }
    const fresh = createUuid();
    memorySessionId = fresh;
    try {
      window.localStorage.setItem(SESSION_KEY, fresh);
    } catch (_) {
      /* noop */
    }
    return fresh;
  }

  function safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch (_) {
      return null;
    }
  }

  function readInlineConfig() {
    const node = doc.getElementById("u4s-config");
    if (!node) return null;
    const payload = node.textContent || node.innerText || "";
    if (!payload.trim()) return null;
    return safeJsonParse(payload.trim());
  }

  function selectEndpoint(source) {
    if (!source || typeof source !== "object") return "";
    return (
      source.fnUrl ||
      source.fn_url ||
      source.endpoint ||
      source.apiEndpoint ||
      source.api_endpoint ||
      source.apiUrl ||
      source.api_url ||
      ""
    );
  }

  function readAttrEndpoint(node) {
    if (!node || typeof node.getAttribute !== "function") return "";
    return node.getAttribute("data-u4s-fn-url") || "";
  }

  function readQueryEndpoint() {
    try {
      const url = new URL(window.location.href);
      return url.searchParams.get("fnUrl") || url.searchParams.get("endpoint") || "";
    } catch (_) {
      return "";
    }
  }

  function resolveEndpoint() {
    const globalConfig = typeof window !== "undefined" ? selectEndpoint(window.__U4S_CONFIG__) : "";
    const inlineConfig = selectEndpoint(readInlineConfig());
    const attrHtml = readAttrEndpoint(root);
    const attrBody = readAttrEndpoint(body);
    const query = readQueryEndpoint();

    return (
      [globalConfig, inlineConfig, attrHtml, attrBody, query, DEFAULT_ENDPOINT]
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .find(Boolean) || DEFAULT_ENDPOINT
    );
  }

  const ENDPOINT = resolveEndpoint();
  const sessionId = ensureSessionId();
  const history = loadHistory();

  // ===== Авто-рост iframe =====

  let resizeTimer = null;

  function computeDocumentHeight() {
    return (
      Math.max(
        body.scrollHeight,
        root.scrollHeight,
        body.offsetHeight,
        root.offsetHeight,
        body.clientHeight,
        root.clientHeight,
      ) + EXTRA_PADDING
    );
  }

  function postSizeNow() {
    try {
      window.parent.postMessage({ type: SIZE_EVENT, height: computeDocumentHeight() }, "*");
    } catch (_) {
      /* noop */
    }
  }

  function schedulePostSize() {
    if (resizeTimer) {
      clearTimeout(resizeTimer);
    }
    resizeTimer = setTimeout(() => {
      window.requestAnimationFrame(postSizeNow);
    }, POST_DEBOUNCE_MS);
  }

  try {
    const resizeObserver = new ResizeObserver(schedulePostSize);
    resizeObserver.observe(root);
    resizeObserver.observe(elements.scroll);
  } catch (_) {
    try {
      const mutationObserver = new MutationObserver(schedulePostSize);
      mutationObserver.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      });
    } catch (__) {
      /* noop */
    }
  }

  window.addEventListener("load", schedulePostSize);
  window.addEventListener("resize", schedulePostSize);

  // страховочные пинги размера
  (function pulseSize() {
    let remaining = BURST_PULSES;
    const interval = setInterval(() => {
      schedulePostSize();
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(interval);
      }
    }, BURST_INTERVAL_MS);
  })();

  // ===== Сообщения =====

  const sanitizeMap = Object.freeze({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  });

  function sanitize(text) {
    return (text || "").replace(/[&<>"']/g, (char) => sanitizeMap[char] || char);
  }

  function formatMessage(text) {
    return sanitize(text).replace(/\n/g, "<br>");
  }

  function createTimeStamp() {
    return timeFormatter.format(new Date());
  }

  function renderMessage({ role, text, timestamp }) {
    const wrapper = doc.createElement("div");
    wrapper.className = `u4s-msg ${role === "me" ? "u4s-me" : "u4s-bot"}`;
    wrapper.dataset.role = role;
    wrapper.dataset.raw = text;
    const stamp = timestamp || createTimeStamp();
    wrapper.dataset.timestamp = stamp;
    wrapper.innerHTML = formatMessage(text);

    const timeNode = doc.createElement("div");
    timeNode.className = "u4s-time";
    timeNode.textContent = stamp;

    elements.scroll.append(wrapper, timeNode);
    scrollToEnd();
    schedulePostSize();
    return wrapper;
  }

  function renderTyping() {
    const row = doc.createElement("div");
    row.className = "u4s-typing";
    row.innerHTML = '<div class="u4s-dot"></div><div class="u4s-dot"></div><div class="u4s-dot"></div>';
    elements.scroll.appendChild(row);
    scrollToEnd();
    schedulePostSize();
    return row;
  }

  function removeTyping(node) {
    if (node && node.parentNode) {
      node.parentNode.removeChild(node);
      schedulePostSize();
    }
  }

  function scrollToEnd() {
    elements.scroll.scrollTop = elements.scroll.scrollHeight + 999;
  }

  function loadHistory() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }

  function persistHistory() {
    const messages = Array.from(elements.scroll.querySelectorAll(".u4s-msg")).map((node) => ({
      role: node.dataset.role === "me" ? "me" : "bot",
      text: node.dataset.raw || node.textContent || "",
      timestamp: node.dataset.timestamp || undefined,
    }));
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (_) {
      /* noop */
    }
    schedulePostSize();
  }

  function collectHistory(limit) {
    const nodes = Array.from(elements.scroll.querySelectorAll(".u4s-msg"));
    const items = nodes.map((node) => ({
      role: node.dataset.role === "me" ? "user" : "assistant",
      text: node.dataset.raw || node.textContent || "",
    }));
    if (typeof limit === "number" && Number.isFinite(limit)) {
      const normalized = Math.max(0, Math.floor(limit));
      return normalized > 0 ? items.slice(-normalized) : [];
    }
    return items;
  }

  function clearChatHistory() {
    elements.scroll.replaceChildren();
    persistHistory();
  }

  function restoreHistory(messages) {
    elements.scroll.replaceChildren();
    if (Array.isArray(messages)) {
      messages.forEach((message) => {
        if (!message || typeof message.text !== "string") return;
        renderMessage({
          role: message.role === "me" ? "me" : "bot",
          text: message.text,
          timestamp: message.timestamp,
        });
      });
    }
    persistHistory();
  }

  function appendBotMessage(text) {
    renderMessage({ role: "bot", text });
    persistHistory();
  }

  function appendUserMessage(text) {
    renderMessage({ role: "me", text });
    persistHistory();
  }

  // восстановление истории
  if (history.length > 0) {
    history.forEach((message) => {
      if (!message || typeof message.text !== "string") return;
      renderMessage({
        role: message.role === "me" ? "me" : "bot",
        text: message.text,
        timestamp: message.timestamp,
      });
    });
  } else {
    appendBotMessage(INITIAL_GREETING);
  }

  function toggleChat(open) {
    const shouldOpen = open ?? elements.chat.getAttribute("data-open") !== "true";
    elements.chat.setAttribute("data-open", shouldOpen ? "true" : "false");
    elements.chat.style.display = shouldOpen ? "flex" : "none";
    if (shouldOpen) {
      setTimeout(() => {
        elements.input.focus();
      }, 50);
    }
    schedulePostSize();
  }

  elements.fab.addEventListener("click", () => toggleChat(true));
  elements.fab.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      toggleChat(true);
    }
  });

  elements.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = (elements.input.value || "").trim();
    if (!text) return;

    appendUserMessage(text);
    elements.input.value = "";
    elements.input.focus();

    const typingNode = renderTyping();
    const payload = { question: text, sessionId };
    const recentHistory = collectHistory(4);
    if (recentHistory.length > 0) {
      payload.history = recentHistory;
    }
    try {
      const response = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json().catch(() => ({}));
      removeTyping(typingNode);

      const answer = [data?.answer, data?.message, data?.response].find((value) => typeof value === "string")
        || "Извините, сейчас не могу ответить. Попробуйте позже.";
      appendBotMessage(answer);
    } catch (error) {
      removeTyping(typingNode);
      appendBotMessage(CONNECTION_ERROR_MESSAGE);
      console.error("U4S widget fetch error", error);
    }
  });

  elements.reset.addEventListener("click", async () => {
    if (elements.reset.disabled) return;
    elements.reset.disabled = true;

    const previousMessages = loadHistory();
    const lastKnownHistory = Array.isArray(previousMessages)
      ? previousMessages.slice(-4).map((message) => ({
          role: message?.role === "me" ? "user" : "assistant",
          text: typeof message?.text === "string" ? message.text : "",
        }))
      : [];

    clearChatHistory();
    const typingNode = renderTyping();

    const resetPayload = { reset: true, action: "reset", sessionId };
    if (lastKnownHistory.length > 0) {
      resetPayload.history = lastKnownHistory;
    }

    try {
      const response = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(resetPayload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      removeTyping(typingNode);
      appendBotMessage(INITIAL_GREETING);
    } catch (error) {
      removeTyping(typingNode);
      restoreHistory(previousMessages);
      appendBotMessage(RESET_ERROR_MESSAGE);
      console.error("U4S widget reset error", error);
    } finally {
      elements.reset.disabled = false;
    }
  });

  function autoOpenFromQuery() {
    try {
      if (new URL(window.location.href).searchParams.get("chat") === "open") {
        toggleChat(true);
      }
    } catch (_) {
      /* noop */
    }
  }

  autoOpenFromQuery();
})();
