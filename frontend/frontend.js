(() => {
  function resolveDefaultEndpoint() {
    return "/api/chat";
  }

  const DEFAULT_ENDPOINT = resolveDefaultEndpoint();
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
  let activeAbortController = null;
  let inMemoryMessages = [];
  function newSessionId() {
    if (typeof window !== "undefined" && window.crypto && typeof crypto.randomUUID === "function") {
      try {
        return crypto.randomUUID();
      } catch (_) {
        /* noop */
      }
    }

    return "sid_" + Date.now() + "_" + Math.random().toString(16).slice(2);
  }

  function getSessionId() {
    try {
      return window.localStorage.getItem(SESSION_KEY) || "";
    } catch (_) {
      return "";
    }
  }

  function setSessionId(id) {
    memorySessionId = (id || "").trim();
    try {
      window.localStorage.setItem(SESSION_KEY, memorySessionId);
    } catch (_) {
      /* noop */
    }
    return memorySessionId;
  }

  function ensureSessionId() {
    if (memorySessionId) {
      return memorySessionId;
    }
    const existing = getSessionId();
    if (existing) {
      memorySessionId = existing;
      return existing;
    }
    const fresh = newSessionId();
    setSessionId(fresh);
    return fresh;
  }

  function resetSessionId() {
    const sid = newSessionId();
    setSessionId(sid);
    return sid;
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
  ensureSessionId();
  const history = loadHistory();
  inMemoryMessages = Array.isArray(history) ? [...history] : [];

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

  function renderMessageText(container, text) {
    container.textContent = "";

    const urlRegex = /(https?:\/\/[^\s]+)/g;
    let lastIndex = 0;
    let match;
    let hasBookingLink = false;

    while ((match = urlRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        container.appendChild(doc.createTextNode(text.slice(lastIndex, match.index)));
      }

      const url = match[0];

      const anchor = doc.createElement("a");
      anchor.href = url;
      anchor.textContent = url;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      anchor.className = "chat-link";

      container.appendChild(anchor);

      if (url.startsWith("https://usadba4.ru/bronirovanie/")) {
        hasBookingLink = true;
      }

      lastIndex = urlRegex.lastIndex;
    }

    if (lastIndex < text.length) {
      container.appendChild(doc.createTextNode(text.slice(lastIndex)));
    }

    if (hasBookingLink) {
      const btnWrap = doc.createElement("div");
      btnWrap.className = "booking-btn-wrap";

      const btn = doc.createElement("a");
      btn.href = "https://usadba4.ru/bronirovanie/";
      btn.target = "_blank";
      btn.rel = "noopener noreferrer";
      btn.textContent = "Забронировать";
      btn.className = "booking-btn";

      btnWrap.appendChild(btn);
      container.appendChild(btnWrap);
    }
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

    const messageBody = doc.createElement("div");
    messageBody.className = "chat-message";
    renderMessageText(messageBody, text);

    wrapper.appendChild(messageBody);

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
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(inMemoryMessages));
    } catch (_) {
      /* noop */
    }
    schedulePostSize();
  }

  function collectHistory(limit) {
    const items = inMemoryMessages.map((message) => ({
      role: message.role === "me" ? "user" : "assistant",
      text: message.text,
    }));
    if (typeof limit === "number" && Number.isFinite(limit)) {
      const normalized = Math.max(0, Math.floor(limit));
      return normalized > 0 ? items.slice(-normalized) : [];
    }
    return items;
  }

  function clearChatHistory() {
    inMemoryMessages = [];
    elements.scroll.replaceChildren();
    persistHistory();
  }

  function restoreHistory(messages) {
    inMemoryMessages = Array.isArray(messages) ? messages.filter((message) => typeof message?.text === "string") : [];
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
    const node = renderMessage({ role: "bot", text });
    const timestamp = node?.dataset?.timestamp || createTimeStamp();
    inMemoryMessages.push({ role: "bot", text, timestamp });
    persistHistory();
  }

  function appendUserMessage(text) {
    const node = renderMessage({ role: "me", text });
    const timestamp = node?.dataset?.timestamp || createTimeStamp();
    inMemoryMessages.push({ role: "me", text, timestamp });
    persistHistory();
  }

  function resetChat() {
    try {
      activeAbortController?.abort();
    } catch (_) {
      /* noop */
    }
    activeAbortController = null;

    resetSessionId();

    inMemoryMessages = [];
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* noop */
    }
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* noop */
    }

    elements.scroll.replaceChildren();
    const typingNodes = doc.querySelectorAll(".u4s-typing");
    typingNodes.forEach((node) => {
      if (node && node.parentNode) {
        node.parentNode.removeChild(node);
      }
    });

    if (elements.input) {
      elements.input.value = "";
      elements.input.focus();
    }

    appendBotMessage(INITIAL_GREETING);
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
    const controller = new AbortController();
    activeAbortController = controller;
    const requestSessionId = ensureSessionId();
    const payload = { message: text, question: text, sessionId: requestSessionId };
    const recentHistory = collectHistory(4);
    if (recentHistory.length > 0) {
      payload.history = recentHistory;
    }
    try {
      const response = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json().catch(() => ({}));
      if (requestSessionId !== getSessionId()) {
        removeTyping(typingNode);
        return;
      }

      removeTyping(typingNode);

      const answer = [data?.answer, data?.message, data?.response].find((value) => typeof value === "string")
        || "Извините, сейчас не могу ответить. Попробуйте позже.";
      appendBotMessage(answer);
    } catch (error) {
      if (error?.name === "AbortError") {
        removeTyping(typingNode);
        return;
      }

      removeTyping(typingNode);
      appendBotMessage(CONNECTION_ERROR_MESSAGE);
      console.error("U4S widget fetch error", error);
    } finally {
      if (activeAbortController === controller) {
        activeAbortController = null;
      }
    }
  });

  doc.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-action='reset-chat'], #u4s-reset, .reset-chat-btn");
    if (!btn) return;
    event.preventDefault();
    resetChat();
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
