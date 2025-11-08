(function(){
  const FN_URL = "https://ai-chatbot-u4s-karinausadba.amvera.io/api/chat"; // ← эндпоинт FastAPI

  const $fab = document.getElementById('u4s-fab');
  const $chat = document.getElementById('u4s-chat');
  const $scroll = document.getElementById('u4s-scroll');
  const $form = document.getElementById('u4s-inputbar');
  const $q = document.getElementById('u4s-q');

  const KEY = 'u4s_history_v1';
  let history = [];
  try{ history = JSON.parse(localStorage.getItem(KEY) || '[]'); }catch(_){}

  // ==== Авто-рост: отправка высоты родителю (Flexbe) ====
  const POST_TYPE = 'u4s-iframe-size';
  const EXTRA_PADDING = 12;               // небольшой запас, чтобы не обрезало тени/бордеры
  const POST_DEBOUNCE_MS = 60;
  let _postTimer = null;

  function getDocHeight(){
    const b = document.body, d = document.documentElement;
    return Math.max(
      b.scrollHeight, d.scrollHeight,
      b.offsetHeight, d.offsetHeight,
      b.clientHeight, d.clientHeight
    ) + EXTRA_PADDING;
  }

  function postSizeNow(){
    try {
      parent.postMessage({ type: POST_TYPE, height: getDocHeight() }, '*'); // origin фильтруется на стороне родителя
    } catch(_) {}
  }

  function postSize(){
    if (_postTimer) clearTimeout(_postTimer);
    _postTimer = setTimeout(()=> {
      // ждём кадр, чтобы DOM дорисовался
      requestAnimationFrame(postSizeNow);
    }, POST_DEBOUNCE_MS);
  }

  // наблюдатели за изменением размеров/DOM
  try {
    const ro = new ResizeObserver(postSize);
    ro.observe(document.documentElement);
    ro.observe($scroll);
  } catch(_) {
    // фолбек: MutationObserver
    try {
      const mo = new MutationObserver(postSize);
      mo.observe(document.documentElement, { childList:true, subtree:true, attributes:true, characterData:true });
    } catch(__) {}
  }

  window.addEventListener('load', postSize);
  window.addEventListener('resize', postSize);

  // ==== Инициализация истории ====
  history.forEach(addBubble);
  scrollToEnd();
  postSize();

  function toggleChat(open){
    const show = open ?? ($chat.style.display==='none' || !$chat.style.display);
    $chat.style.display = show ? 'flex' : 'none';
    if (show) setTimeout(()=> $q && $q.focus(), 50);
    postSize();
  }
  $fab.addEventListener('click', ()=>toggleChat(true));
  $fab.addEventListener('keydown', e=>{ if(e.key==='Enter' || e.key===' ') toggleChat(true); });

  if(history.length===0){
    addBubble({role:'bot', text:'Здравствуйте! Я помогу с вопросами про проживание, ресторан «Калина Красная», баню, контакты и др.'});
    persist();
  }

  $form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const text = ($q.value || '').trim();
    if(!text) return;
    addBubble({role:'me', text});
    $q.value=''; $q.focus();
    persist();

    const typing = addTyping();
    try{
      const res = await fetch(FN_URL, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ question: text })
      });
      const data = await res.json().catch(()=> ({}));
      removeTyping(typing);

      let answer = data?.answer || data?.message || data?.response || '';
      if(!answer){ answer = 'Извините, сейчас не могу ответить. Попробуйте позже.'; }
      addBubble({role:'bot', text: answer});
      persist();
    }catch(err){
      removeTyping(typing);
      addBubble({role:'bot', text:'Упс! Не удалось связаться с сервером.'});
      persist();
    }
  });

  function addBubble({role, text}){
    const wrap = document.createElement('div');
    wrap.className='u4s-msg '+(role==='me'?'u4s-me':'u4s-bot');
    wrap.innerHTML = sanitize(text).replace(/\n/g,'<br>');
    $scroll.appendChild(wrap);

    const time = document.createElement('div');
    time.className='u4s-time';
    time.textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    $scroll.appendChild(time);

    scrollToEnd();
    postSize();
  }

  function addTyping(){
    const row = document.createElement('div');
    row.className='u4s-typing';
    row.innerHTML = '<div class="u4s-dot"></div><div class="u4s-dot"></div><div class="u4s-dot"></div>';
    $scroll.appendChild(row);
    scrollToEnd();
    postSize();
    return row;
  }

  function removeTyping(node){
    if(node && node.parentNode) node.parentNode.removeChild(node);
    postSize();
  }

  function scrollToEnd(){
    $scroll.scrollTop = $scroll.scrollHeight + 999;
  }

  function persist(){
    const msgs = [...document.querySelectorAll('.u4s-msg')]
      .map(n=>({role:n.classList.contains('u4s-me')?'me':'bot', text:n.innerText}));
    try{ localStorage.setItem(KEY, JSON.stringify(msgs)); }catch(_){}
    postSize();
  }

  function sanitize(s){
    return (s||'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]));
  }

  // авто-открытие ?chat=open
  try{ if(new URL(location.href).searchParams.get('chat')==='open') toggleChat(true); }catch(_){}

  // страховочные пинги размера в первые 5 сек
  (function burst(){
    let left = 12;
    const iv = setInterval(()=> {
      postSize();
      if(--left <= 0) clearInterval(iv);
    }, 400);
  })();
})();

