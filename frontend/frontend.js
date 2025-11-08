(() => {
  const DEFAULT_ENDPOINT = "https://ai-chatbot-u4s-karinausadba.amvera.io/api/chat";
  const STORAGE_KEY = "u4s_history_v1";
  const SIZE_EVENT = "u4s-iframe-size";
  const EXTRA_PADDING = 12;
  const POST_DEBOUNCE_MS = 60;
  const BURST_PULSES = 12;
  const BURST_INTERVAL_MS = 400;

  const doc = document;
  const root = doc.documentElement;
  const body = doc.body;

  const elements = {
    fab: doc.getElementById("u4s-fab"),
    chat: doc.getElementById("u4s-chat"),
    scroll: doc.getElementById("u4s-scroll"),
    form: doc.getElementById("u4s-inputbar"),
    input: doc.getElementById("u4s-q"),
  };

  if (!elements.fab || !elements.chat || !elements.scroll || !elements.form || !elements.input) {
    console.warn("U4S widget: required DOM nodes are missing");
    return;
  }

  const timeFormatter = new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" });

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
    appendBotMessage(
      "Здравствуйте! Я помогу с вопросами про проживание, ресторан «Калина Красная», баню, контакты и др.",
    );
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
    try {
      const response = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
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
      appendBotMessage("Упс! Не удалось связаться с сервером.");
      console.error("U4S widget fetch error", error);
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
