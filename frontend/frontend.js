(function(){
  const FN_URL = "https://ai-chatbot-u4s-karinausadba.amvera.io"; // ← твоя функция

  const $fab = document.getElementById('u4s-fab');
  const $chat = document.getElementById('u4s-chat');
  const $scroll = document.getElementById('u4s-scroll');
  const $form = document.getElementById('u4s-inputbar');
  const $q = document.getElementById('u4s-q');

  const KEY = 'u4s_history_v1';
  let history = [];
  try{ history = JSON.parse(localStorage.getItem(KEY) || '[]'); }catch(_){}
  history.forEach(addBubble); scrollToEnd();

  function toggleChat(open){
    const show = open ?? ($chat.style.display==='none' || !$chat.style.display);
    $chat.style.display = show ? 'flex' : 'none';
    if (show) setTimeout(()=> $q.focus(), 50);
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
  }
  function addTyping(){
    const row = document.createElement('div'); row.className='u4s-typing';
    row.innerHTML = '<div class="u4s-dot"></div><div class="u4s-dot"></div><div class="u4s-dot"></div>';
    $scroll.appendChild(row); scrollToEnd();
    return row;
  }
  function removeTyping(node){ if(node && node.parentNode) node.parentNode.removeChild(node); }
  function scrollToEnd(){ $scroll.scrollTop = $scroll.scrollHeight+999; }
  function persist(){
    const msgs = [...document.querySelectorAll('.u4s-msg')].map(n=>({role:n.classList.contains('u4s-me')?'me':'bot', text:n.innerText}));
    localStorage.setItem(KEY, JSON.stringify(msgs));
  }
  function sanitize(s){ return (s||'').replace(/[&<>"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m])); }

  // авто-открытие ?chat=open
  try{ if(new URL(location.href).searchParams.get('chat')==='open') toggleChat(true); }catch(_){}
})();
