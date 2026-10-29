// Browser entry point: restore chat/status, then send user actions to FastAPI.
// Chat and plan history are saved by the server; sessionStorage keeps only the active chat ID.
// This file renders answers, source links and quizzes; retrieval and grading run in Python.
const getElementById = id => document.getElementById(id);
let sessionId = null, busy = false;
let pendingPlanId = null, revisePlanId = null;

// Build elements with textContent so documents and model output cannot inject HTML.
function createElement(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Send JSON to a FastAPI route and return its parsed response.
// Convert failed HTTP responses into readable errors for runUiAction() to display.
async function requestApi(path, body, method = body === undefined ? "GET" : "POST") {
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

// Run one browser action at a time, show elapsed waiting time and restore controls.
// This keeps repeated clicks from sending duplicate requests while a local model is busy.
async function runUiAction(task, errorTarget = getElementById("status"), loading = "Working…") {
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
    document.querySelectorAll("button").forEach(button => button.disabled = Boolean(button.dataset.planId && button.dataset.planId !== pendingPlanId));
  }
}

// Render answer paragraphs, headings, code and bold text using DOM text nodes.
// Model/document text is never inserted as raw HTML, so it cannot create page scripts.
function renderMarkdown(target, text) {
  const blocks = text.split(/(```[\s\S]*?```)/g);
  for (const block of blocks) {
    if (block.startsWith("```")) {
      const pre = createElement("pre");
      pre.append(createElement("code", block.replace(/^```[^\n]*\n?/, "").replace(/```$/, "")));
      target.append(pre);
    } else {
      for (const paragraph of block.split(/\n\s*\n/).filter(Boolean)) {
        if (/^#{1,3} /.test(paragraph)) target.append(createElement("h3", paragraph.replace(/^#{1,3} /, "")));
        else {
          const p = createElement("p");
          paragraph.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(part => {
            if (part.startsWith("**")) p.append(createElement("strong", part.slice(2, -2)));
            else if (part.startsWith("`")) p.append(createElement("code", part.slice(1, -1)));
            else p.append(document.createTextNode(part));
          });
          target.append(p);
        }
      }
    }
  }
}

// Show the source labels and links returned with an answer or graded quiz.
// Validate HTTPS links before creating anchors; display names stay plain text.
function renderSourceLinks(target, rows = []) {
  const group = createElement("div", undefined, "sources");
  for (const [index, row] of rows.entries()) {
    try {
      const url = new URL(row.source_url || row.url);
      if (url.protocol !== "https:") continue;
      const label = /^S\d+$/.test(row.id || "") ? row.id : index + 1;
      const link = createElement("a", `${label}  ${row.title} · ${row.location || ""}`, "source");
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      group.append(link);
    } catch { /* Skip malformed source links rather than making them clickable. */ }
  }
  target.append(group);
}

// Download a portable note with the answer and its original source links.
function downloadAnswerMarkdown(text, refs) {
  const citations = refs.map((row, index) => `${row.id || index + 1}. ${row.title} — ${row.location || ""}\n   ${row.source_url || row.url || ""}`).join("\n");
  const blob = new Blob([text + "\n\n## Sources\n\n" + citations], {type: "text/markdown;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = createElement("a");
  link.href = url;
  link.download = "team-knowledge-note.md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Append a real chat message and retain source links when reopening saved conversations.
function appendChatMessage(role, text, refs = [], quiz = null, plan = null) {
  getElementById("welcome").hidden = true;
  const box = createElement("article", undefined, "message " + role);
  // Display short citation numbers instead of internal chunk IDs.
  let displayText = text;
  refs.forEach((row, index) => {
    displayText = displayText.split("[" + row.id + "]").join("[" + (index + 1) + "]");
  });
  renderMarkdown(box, displayText);
  if (role === "assistant") {
    renderSourceLinks(box, refs);
    const save = createElement("button", "Save as Markdown", "subtle");
    save.onclick = () => downloadAnswerMarkdown(displayText, refs);
    box.append(save);
    if (quiz) {
      const open = createElement("button", "Open quiz", "subtle");
      open.onclick = () => renderQuizQuestions(quiz);
      box.append(open);
    }
    if (plan) {
      const approve = createElement("button", "Approve plan", "primary");
      const revise = createElement("button", "Revise plan", "subtle");
      for (const button of [approve, revise]) {
        button.dataset.planId = plan.plan_id;
        button.disabled = plan.plan_id !== pendingPlanId;
      }
      approve.onclick = () => runUiAction(async () => {
        displayChatResponse(await requestApi("/api/chat", {session_id: sessionId, action: "approve", plan_id: plan.plan_id}));
        await refreshChatHistory();
      }, getElementById("status"), "Running approved plan…");
      revise.onclick = () => {
        revisePlanId = plan.plan_id;
        getElementById("message").placeholder = "Describe what to add, remove, or change in this plan…";
        getElementById("message").focus();
      };
      box.append(approve, revise);
    }
  }
  getElementById("messages").append(box);
  getElementById("conversation").scrollTop = getElementById("conversation").scrollHeight;
  return box;
}

// Fetch saved conversation IDs/titles and rebuild the sidebar open/delete controls.
// The server remains the source of truth for chat history.
async function refreshChatHistory() {
  const chats = await requestApi("/api/sessions");
  getElementById("history").replaceChildren();
  for (const chat of chats.reverse()) {
    const row = createElement("div", undefined, "history-item");
    const open = createElement("button", chat.title, "chat-link" + (chat.session_id === sessionId ? " active" : ""));
    open.onclick = () => runUiAction(() => restoreChatConversation(chat));
    const remove = createElement("button", "×", "delete-chat");
    remove.title = `Delete ${chat.title}`;
    remove.setAttribute("aria-label", `Delete ${chat.title}`);
    remove.onclick = () => deleteChatConversation(chat);
    row.append(open, remove);
    getElementById("history").append(row);
  }
}

// Delete only the selected conversation after a deliberate confirmation.
async function deleteChatConversation(chat) {
  if (!confirm(`Delete "${chat.title}"?`)) return;
  await runUiAction(async () => {
    await requestApi("/api/sessions/" + encodeURIComponent(chat.session_id), undefined, "DELETE");
    if (chat.session_id === sessionId) clearChatForNewConversation();
    await refreshChatHistory();
  }, getElementById("status"), "Deleting chat…");
}

// Load a saved conversation, render its messages and restore the current pending plan ID.
// Only approval controls for that proposal remain active after reopening the chat.
async function restoreChatConversation(chat) {
  const result = await requestApi("/api/sessions/" + encodeURIComponent(chat.session_id));
  sessionId = result.session_id;
  pendingPlanId = result.pending_plan_id || null;
  revisePlanId = null;
  sessionStorage.setItem("activeChatId", sessionId);
  getElementById("messages").replaceChildren();
  getElementById("chat-title").textContent = chat.title;
  for (const message of result.messages) {
    appendChatMessage(message.role, message.content, message.sources || [], message.quiz, message.plan);
  }
  document.body.classList.remove("sidebar-open");
  await refreshChatHistory();
}

// Clear the visible conversation and remember its new ID before the first request.
// A browser refresh can then reopen the same server-side chat while an answer is pending.
function clearChatForNewConversation(id = crypto.randomUUID()) {
  sessionId = id;
  pendingPlanId = null;
  revisePlanId = null;
  sessionStorage.setItem("activeChatId", sessionId);
  getElementById("messages").replaceChildren();
  getElementById("welcome").hidden = false;
  getElementById("chat-title").textContent = "New conversation";
  getElementById("status").textContent = "";
  document.body.classList.remove("sidebar-open");
  getElementById("message").focus();
}

// Display readiness as a small footnote; failed status requests do not hide the conversation.
async function refreshIndexStatus() {
  try {
    const status = await requestApi("/api/status");
    getElementById("knowledge-status").textContent = status.ready ? "● Knowledge ready" : "No documents indexed";
    getElementById("sync-status").textContent = status.synced_at ? "Synced " + new Date(status.synced_at).toLocaleString() : "";
  } catch {
    getElementById("knowledge-status").textContent = "Knowledge status unavailable";
  }
}

// Display the score, correct choices, explanations and sources returned after submission.
// Python grades the saved answer key; this function only presents the result.
function renderQuizResult(result, answers) {
  const root = getElementById("quiz-body");
  root.replaceChildren(createElement("h3", `Score: ${result.score} / ${result.total}`));
  result.questions.forEach((question, index) => {
    const section = createElement("section", undefined, "result");
    section.append(createElement("h3", question.question),
      createElement("p", "Your answer: " + question.options[answers[index]]),
      createElement("p", "Correct answer: " + question.options[question.correct_index]),
      createElement("p", question.explanation));
    renderSourceLinks(section, (result.sources || []).filter(row => question.source_ids.includes(row.id)));
    root.append(section);
  });
  root.append(createElement("p", result.kind === "daily" ? "Your answer is saved. A new challenge arrives tomorrow." : "Quiz completed."));
  const another = createElement("button", "Create another quiz", "subtle");
  another.onclick = insertQuizCommand;
  root.append(another);
}

// Build the quiz form from question text and choices received from the server.
// Submit selected option indexes to the daily or custom-quiz route for grading.
function renderQuizQuestions(quiz) {
  const daily = quiz.kind === "daily";
  getElementById("quiz-title").textContent = daily ? "Daily Challenge" : "Practice quiz";
  getElementById("quiz-subtitle").textContent = daily ? "One question from your team's knowledge" : "Test what you have learned";
  getElementById("quiz-error").textContent = "";
  const dialog = getElementById("quiz-dialog");
  if (!dialog.open) dialog.showModal();
  if (quiz.submitted) {
    renderQuizResult(quiz, quiz.answers);
    return;
  }
  const root = getElementById("quiz-body");
  root.replaceChildren(createElement("span", quiz.difficulty + " · " + quiz.topic, "badge"));
  const form = createElement("form");
  quiz.questions.forEach((question, index) => {
    const field = createElement("fieldset");
    field.append(createElement("legend", question.question));
    question.options.forEach((option, choice) => {
      const label = createElement("label", undefined, "option");
      const input = createElement("input");
      input.type = "radio";
      input.name = "q" + index;
      input.value = choice;
      input.required = true;
      label.append(input, document.createTextNode(option));
      field.append(label);
    });
    form.append(field);
  });
  const actions = createElement("div", undefined, "quiz-actions");
  actions.append(createElement("button", "Check answer", "primary"));
  form.append(actions);
  form.onsubmit = event => {
    event.preventDefault();
    const data = new FormData(form);
    const answers = quiz.questions.map((_, index) => Number(data.get("q" + index)));
    const path = daily ? `/api/daily-challenge/${quiz.day}/submit` : `/api/quizzes/${quiz.quiz_id}/submit`;
    runUiAction(async () => {
      const result = await requestApi(path, {answers});
      renderQuizResult({...result, kind: quiz.kind}, result.answers || answers);
    }, getElementById("quiz-error"), "Checking your answer…");
  };
  root.append(form);
}

// Put the exact supported command in the composer so a new quiz needs no settings page.
function insertQuizCommand() {
  getElementById("quiz-dialog").close();
  getElementById("message").value = '/quiz "your topic" medium 5';
  getElementById("message").focus();
  getElementById("message").setSelectionRange(7, 17);
}

// Update the active conversation and render an answer, proposal or generated quiz.
// Remember the newest plan ID so older proposal buttons cannot approve a different plan.
function displayChatResponse(result) {
  sessionId = result.session_id;
  pendingPlanId = result.pending_plan_id || null;
  revisePlanId = null;
  sessionStorage.setItem("activeChatId", sessionId);
  getElementById("message").placeholder = "Ask about your research papers…";
  document.querySelectorAll("button[data-plan-id]").forEach(button => button.disabled = button.dataset.planId !== pendingPlanId);
  if (result.kind === "quiz") {
    appendChatMessage("assistant", "Quiz created: " + result.topic, [], result);
    renderQuizQuestions(result);
  } else appendChatMessage("assistant", result.answer, result.sources, null, result.plan);
}


// Send composer text, chat ID and any plan-revision action to /api/chat.
// Restore failed input for retry; on success display the response and refresh sidebar history.
async function submitChatMessage(event) {
  event.preventDefault();
  const message = getElementById("message").value.trim();
  if (!message || busy) return;
  await runUiAction(async () => {
    // The ID is already in sessionStorage before the request starts. If the user
    // refreshes while Gemma is working, the browser returns to this same chat.
    const bubble = appendChatMessage("human", message);
    getElementById("message").value = "";
    try {
      const review = revisePlanId ? {action: "revise", plan_id: revisePlanId} : {};
      const result = await requestApi("/api/chat", {message, session_id: sessionId, ...review});
      displayChatResponse(result);
      if (getElementById("chat-title").textContent === "New conversation") getElementById("chat-title").textContent = message.slice(0, 70);
    } catch (error) {
      bubble.remove();
      getElementById("message").value = message;
      throw error;
    }
    await refreshChatHistory();
  }, getElementById("status"), "Searching your team's knowledge…");
}

// Bind controls once after the deferred script has loaded the page.
getElementById("chat-form").onsubmit = submitChatMessage;
getElementById("new-chat").onclick = () => clearChatForNewConversation();
getElementById("quiz-command").onclick = insertQuizCommand;
getElementById("close-quiz").onclick = () => getElementById("quiz-dialog").close();
getElementById("toggle-sidebar").onclick = () => document.body.classList.toggle("sidebar-open");
getElementById("daily-button").onclick = () => {
  getElementById("quiz-title").textContent = "Daily Challenge";
  getElementById("quiz-subtitle").textContent = "One question from your team's knowledge";
  getElementById("quiz-body").replaceChildren();
  getElementById("quiz-dialog").showModal();
  runUiAction(async () => renderQuizQuestions(await requestApi("/api/daily-challenge")), getElementById("quiz-error"), "Preparing today's challenge…");
};
getElementById("message").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) getElementById("chat-form").requestSubmit();
  }
};

// On page load, restore the active chat if available and fetch history/index status.
// If the saved chat no longer exists, prepare a new conversation instead.
async function restoreChatAndLoadStatus() {
  const savedId = sessionStorage.getItem("activeChatId");
  if (savedId) {
    try {
      await restoreChatConversation({session_id: savedId, title: "Conversation"});
    } catch {
      clearChatForNewConversation(savedId);
    }
  } else {
    clearChatForNewConversation();
  }
  await refreshChatHistory();
  await refreshIndexStatus();
}

restoreChatAndLoadStatus().catch(error => getElementById("status").textContent = error.message);
