// Small DOM helpers and explicit event handlers keep the client framework-free.
const $ = id => document.getElementById(id);
let sessionId = null, busy = false;

// Build elements with textContent so documents and model output cannot inject HTML.
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Call the shared API and expose only its readable error message.
async function api(path, body, method = body === undefined ? "GET" : "POST") {
  const response = await fetch(path, {
    method,
    headers: {"Content-Type": "application/json"},
    ...(body === undefined ? {} : {body: JSON.stringify(body)})
  });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    throw Error(typeof data.detail === "string" ? data.detail : "Please check your input and try again.");
  }
  return data;
}

// Serialize user actions and always restore controls, including after a failed request.
async function run(task, errorTarget = $("status"), loading = "Working…") {
  if (busy) return;
  busy = true;
  const started = Date.now();
  const updateElapsed = () => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    errorTarget.textContent = `${loading} ${seconds}s`;
  };
  updateElapsed();
  errorTarget.classList.add("working");
  const timer = setInterval(updateElapsed, 1000);
  let completed = false;
  document.querySelectorAll("button:not(#close-quiz)").forEach(button => button.disabled = true);
  try {
    await task();
    completed = true;
  } catch (error) {
    errorTarget.textContent = error.message;
  } finally {
    clearInterval(timer);
    errorTarget.classList.remove("working");
    if (completed) {
      const seconds = Math.max(1, Math.round((Date.now() - started) / 1000));
      errorTarget.textContent = `Finished in ${seconds}s`;
      setTimeout(() => {
        if (errorTarget.textContent.startsWith("Finished in ")) errorTarget.textContent = "";
      }, 5000);
    }
    busy = false;
    document.querySelectorAll("button").forEach(button => button.disabled = false);
  }
}

// Render a deliberately small Markdown subset without evaluating raw HTML.
function markdown(target, text) {
  const blocks = text.split(/(```[\s\S]*?```)/g);
  for (const block of blocks) {
    if (block.startsWith("```")) {
      const pre = element("pre");
      pre.append(element("code", block.replace(/^```[^\n]*\n?/, "").replace(/```$/, "")));
      target.append(pre);
    } else {
      for (const paragraph of block.split(/\n\s*\n/).filter(Boolean)) {
        if (/^#{1,3} /.test(paragraph)) target.append(element("h3", paragraph.replace(/^#{1,3} /, "")));
        else {
          const p = element("p");
          paragraph.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(part => {
            if (part.startsWith("**")) p.append(element("strong", part.slice(2, -2)));
            else if (part.startsWith("`")) p.append(element("code", part.slice(1, -1)));
            else p.append(document.createTextNode(part));
          });
          target.append(p);
        }
      }
    }
  }
}

// Allow only HTTPS citations; source text never controls HTML attributes directly.
function sources(target, rows = []) {
  const group = element("div", undefined, "sources");
  for (const [index, row] of rows.entries()) {
    try {
      const url = new URL(row.source_url || row.url);
      if (url.protocol !== "https:") continue;
      const link = element("a", `${index + 1}  ${row.title} · ${row.location || ""}`, "source");
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      group.append(link);
    } catch { /* Skip malformed source links rather than making them clickable. */ }
  }
  target.append(group);
}

// Download a portable note with the answer and its original source links.
function exportMarkdown(text, refs) {
  const citations = refs.map((row, index) => `${index + 1}. ${row.title} — ${row.location || ""}\n   ${row.source_url || row.url || ""}`).join("\n");
  const blob = new Blob([text + "\n\n## Sources\n\n" + citations], {type: "text/markdown;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = element("a");
  link.href = url;
  link.download = "team-knowledge-note.md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Append a real chat message and retain source links when reopening saved conversations.
function addMessage(role, text, refs = [], quiz = null) {
  $("welcome").hidden = true;
  const box = element("article", undefined, "message " + role);
  // Display short citation numbers instead of internal chunk IDs.
  let displayText = text;
  refs.forEach((row, index) => {
    displayText = displayText.split("[" + row.id + "]").join("[" + (index + 1) + "]");
  });
  markdown(box, displayText);
  if (role === "assistant") {
    sources(box, refs);
    const save = element("button", "Save as Markdown", "subtle");
    save.onclick = () => exportMarkdown(displayText, refs);
    box.append(save);
    if (quiz) {
      const open = element("button", "Open quiz", "subtle");
      open.onclick = () => renderQuiz(quiz);
      box.append(open);
    }
  }
  $("messages").append(box);
  $("conversation").scrollTop = $("conversation").scrollHeight;
  return box;
}

// Refresh chat history from the server instead of keeping another copy in browser storage.
async function refreshHistory() {
  const chats = await api("/api/sessions");
  $("history").replaceChildren();
  for (const chat of chats.reverse()) {
    const row = element("div", undefined, "history-item");
    const open = element("button", chat.title, "chat-link" + (chat.session_id === sessionId ? " active" : ""));
    open.onclick = () => run(() => openChat(chat));
    const remove = element("button", "×", "delete-chat");
    remove.title = `Delete ${chat.title}`;
    remove.setAttribute("aria-label", `Delete ${chat.title}`);
    remove.onclick = () => deleteChat(chat);
    row.append(open, remove);
    $("history").append(row);
  }
}

// Delete only the selected conversation after a deliberate confirmation.
async function deleteChat(chat) {
  if (!confirm(`Delete "${chat.title}"?`)) return;
  await run(async () => {
    await api("/api/sessions/" + encodeURIComponent(chat.session_id), undefined, "DELETE");
    if (chat.session_id === sessionId) startNewChat();
    await refreshHistory();
  }, $("status"), "Deleting chat…");
}

// Reopen a shared conversation with its saved citations and quizzes.
async function openChat(chat) {
  const result = await api("/api/sessions/" + encodeURIComponent(chat.session_id));
  sessionId = result.session_id;
  sessionStorage.setItem("activeChatId", sessionId);
  $("messages").replaceChildren();
  $("chat-title").textContent = chat.title;
  for (const message of result.messages) {
    addMessage(message.role, message.content, message.sources || [], message.quiz);
  }
  document.body.classList.remove("sidebar-open");
  await refreshHistory();
}

// Start a client-identified chat so refreshes reuse the same conversation.
function startNewChat(id = crypto.randomUUID()) {
  sessionId = id;
  sessionStorage.setItem("activeChatId", sessionId);
  $("messages").replaceChildren();
  $("welcome").hidden = false;
  $("chat-title").textContent = "New conversation";
  $("status").textContent = "";
  document.body.classList.remove("sidebar-open");
  $("message").focus();
}

// Display readiness as a small footnote; failed status requests do not hide the conversation.
async function refreshStatus() {
  try {
    const status = await api("/api/status");
    $("knowledge-status").textContent = status.ready ? "● Knowledge ready" : "No documents indexed";
    $("sync-status").textContent = status.synced_at ? "Synced " + new Date(status.synced_at).toLocaleString() : "";
  } catch {
    $("knowledge-status").textContent = "Knowledge status unavailable";
  }
}

// Render the server's graded result; never calculate or ship answer keys before submission.
function renderResult(result, answers) {
  const root = $("quiz-body");
  root.replaceChildren(element("h3", `Score: ${result.score} / ${result.total}`));
  result.questions.forEach((question, index) => {
    const section = element("section", undefined, "result");
    section.append(element("h3", question.question),
      element("p", "Your answer: " + question.options[answers[index]]),
      element("p", "Correct answer: " + question.options[question.correct_index]),
      element("p", question.explanation));
    sources(section, (result.sources || []).filter(row => question.source_ids.includes(row.id)));
    root.append(section);
  });
  root.append(element("p", result.kind === "daily" ? "Your answer is saved. A new challenge arrives tomorrow." : "Quiz completed."));
  const another = element("button", "Create another quiz", "subtle");
  another.onclick = prepareQuiz;
  root.append(another);
}

// Open either the shared daily question or a custom quiz using the same modal.
function renderQuiz(quiz) {
  const daily = quiz.kind === "daily";
  $("quiz-title").textContent = daily ? "Daily Challenge" : "Practice quiz";
  $("quiz-subtitle").textContent = daily ? "One question from your team's knowledge" : "Test what you have learned";
  $("quiz-error").textContent = "";
  const dialog = $("quiz-dialog");
  if (!dialog.open) dialog.showModal();
  if (quiz.submitted) {
    renderResult(quiz, quiz.answers);
    return;
  }
  const root = $("quiz-body");
  root.replaceChildren(element("span", quiz.difficulty + " · " + quiz.topic, "badge"));
  const form = element("form");
  quiz.questions.forEach((question, index) => {
    const field = element("fieldset");
    field.append(element("legend", question.question));
    question.options.forEach((option, choice) => {
      const label = element("label", undefined, "option");
      const input = element("input");
      input.type = "radio";
      input.name = "q" + index;
      input.value = choice;
      input.required = true;
      label.append(input, document.createTextNode(option));
      field.append(label);
    });
    form.append(field);
  });
  const actions = element("div", undefined, "quiz-actions");
  actions.append(element("button", "Check answer", "primary"));
  form.append(actions);
  form.onsubmit = event => {
    event.preventDefault();
    const data = new FormData(form);
    const answers = quiz.questions.map((_, index) => Number(data.get("q" + index)));
    const path = daily ? `/api/daily-challenge/${quiz.day}/submit` : `/api/quizzes/${quiz.quiz_id}/submit`;
    run(async () => {
      const result = await api(path, {answers});
      renderResult({...result, kind: quiz.kind}, result.answers || answers);
    }, $("quiz-error"), "Checking your answer…");
  };
  root.append(form);
}

// Put the exact supported command in the composer so a new quiz needs no settings page.
function prepareQuiz() {
  $("quiz-dialog").close();
  $("message").value = '/quiz "your topic" medium 5';
  $("message").focus();
  $("message").setSelectionRange(7, 17);
}

// Keep failed input available for retry and avoid duplicate user bubbles on errors.
async function sendMessage(event) {
  event.preventDefault();
  const message = $("message").value.trim();
  if (!message || busy) return;
  await run(async () => {
    // The ID is already in sessionStorage before the request starts. If the user
    // refreshes while Gemma is working, the browser returns to this same chat.
    const bubble = addMessage("human", message);
    $("message").value = "";
    try {
      const result = await api("/api/chat", {message, session_id: sessionId});
      sessionId = result.session_id;
      sessionStorage.setItem("activeChatId", sessionId);
      if (result.kind === "quiz") {
        addMessage("assistant", "Quiz created: " + result.topic, [], result);
        renderQuiz(result);
      } else addMessage("assistant", result.answer, result.sources);
      if ($("chat-title").textContent === "New conversation") $("chat-title").textContent = message.slice(0, 70);
    } catch (error) {
      bubble.remove();
      $("message").value = message;
      throw error;
    }
    await refreshHistory();
  }, $("status"), "Searching your team's knowledge…");
}

// Bind controls once after the deferred script has loaded the page.
$("chat-form").onsubmit = sendMessage;
$("new-chat").onclick = () => startNewChat();
$("quiz-command").onclick = prepareQuiz;
$("close-quiz").onclick = () => $("quiz-dialog").close();
$("toggle-sidebar").onclick = () => document.body.classList.toggle("sidebar-open");
$("daily-button").onclick = () => {
  $("quiz-title").textContent = "Daily Challenge";
  $("quiz-subtitle").textContent = "One question from your team's knowledge";
  $("quiz-body").replaceChildren();
  $("quiz-dialog").showModal();
  run(async () => renderQuiz(await api("/api/daily-challenge")), $("quiz-error"), "Preparing today's challenge…");
};
$("message").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) $("chat-form").requestSubmit();
  }
};

// Restore the active conversation after refresh, or prepare one stable new ID.
async function initialize() {
  const savedId = sessionStorage.getItem("activeChatId");
  if (savedId) {
    try {
      await openChat({session_id: savedId, title: "Conversation"});
    } catch {
      startNewChat(savedId);
    }
  } else {
    startNewChat();
  }
  await refreshHistory();
  await refreshStatus();
}

initialize().catch(error => $("status").textContent = error.message);
