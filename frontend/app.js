/**
 * SciChat — Scientific Computing Assistant Frontend
 * Pure Vanilla JavaScript ES6+ implementation
 */

(() => {
  'use strict';

  // Determine Backend API Base URL
  const isLocalBackend = window.location.port === '8000';
  const API_BASE = isLocalBackend ? '' : 'http://localhost:8000';

  // State
  const state = {
    currentSessionId: null,
    sessions: [],
    messages: [],
    jobs: [],
    status: {
      online: false,
      model: 'qwen2.5-coder:7b',
      vectordbChunks: 0,
      kgEntities: {},
    },
    isGenerating: false,
    activeDomain: 'all',
    chatMode: 'rag', // 'rag' or 'baseline'
  };

  // DOM Elements Cache
  const elements = {
    // Top Nav
    statusIndicator: document.getElementById('status-indicator'),
    backendStatusText: document.getElementById('backend-status-text'),
    modelBadge: document.getElementById('model-badge'),
    chunksStat: document.getElementById('chunks-stat'),
    dockerStatusPill: document.getElementById('docker-status-pill'),
    dockerStatusText: document.getElementById('docker-status-text'),
    reindexBtn: document.getElementById('reindex-btn'),
    toggleSidebarBtn: document.getElementById('toggle-sidebar-btn'),
    toggleDrawerBtn: document.getElementById('toggle-drawer-btn'),
    closeDrawerBtn: document.getElementById('close-drawer-btn'),
    jobsBadgeDot: document.getElementById('jobs-badge-dot'),

    // Sidebar
    sidebar: document.getElementById('sidebar'),
    newChatBtn: document.getElementById('new-chat-btn'),
    domainFilters: document.getElementById('domain-filters'),
    sessionList: document.getElementById('session-list'),
    sessionCountLabel: document.getElementById('session-count-label'),

    // Chat Stage
    chatStage: document.getElementById('chat-stage'),
    messagesContainer: document.getElementById('messages-container'),
    heroScreen: document.getElementById('hero-screen'),
    messageFeed: document.getElementById('message-feed'),
    typingIndicator: document.getElementById('typing-indicator'),

    // Input Dock
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    clearChatBtn: document.getElementById('clear-chat-btn'),

    // Inspector Drawer
    inspectorDrawer: document.getElementById('inspector-drawer'),
    drawerTabs: document.querySelectorAll('.drawer-tab'),
    drawerPanes: {
      jobs: document.getElementById('pane-jobs'),
      knowledge: document.getElementById('pane-knowledge'),
      cheatsheet: document.getElementById('pane-cheatsheet'),
    },
    jobsList: document.getElementById('jobs-list'),
    refreshJobsBtn: document.getElementById('refresh-jobs-btn'),
    statVectorCount: document.getElementById('stat-vector-count'),
    statKgEntities: document.getElementById('stat-kg-entities'),
    triggerReindexBtn: document.getElementById('trigger-reindex-btn'),

    // Toast
    toastContainer: document.getElementById('toast-container'),
  };

  // =========================================================================
  // TOAST NOTIFICATIONS
  // =========================================================================
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.innerHTML = `<span class="toast-icon">${icon}</span><span class="toast-msg">${escapeHtml(message)}</span>`;
    
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // =========================================================================
  // API HELPERS
  // =========================================================================
  async function apiGet(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  }

  async function apiPost(endpoint, body) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  }

  // =========================================================================
  // SYSTEM HEALTH & STATUS
  // =========================================================================
  async function checkBackendHealth() {
    try {
      const data = await apiGet('/api/status').catch(() => apiGet('/'));
      state.status.online = true;
      state.status.model = data.model || 'qwen2.5-coder:7b';
      state.status.vectordbChunks = data.vectordb_chunks || 0;

      elements.statusIndicator.className = 'status-dot online pulse';
      elements.backendStatusText.textContent = 'Backend Active';
      elements.modelBadge.textContent = state.status.model;
      elements.chunksStat.textContent = state.status.vectordbChunks;
      elements.statVectorCount.textContent = state.status.vectordbChunks;
    } catch (err) {
      state.status.online = false;
      elements.statusIndicator.className = 'status-dot offline';
      elements.backendStatusText.textContent = 'Offline';
    }
  }

  async function checkDockerStatus() {
    if (!elements.dockerStatusText) return;
    try {
      const data = await apiGet('/api/docker/status');
      if (data.docker_available && data.image_available) {
        elements.dockerStatusText.textContent = 'Docker: Ready';
        elements.dockerStatusPill.className = 'stat-pill success';
        elements.dockerStatusPill.title = `Isolated Docker Sandbox Ready (${data.image_name})`;
      } else if (data.docker_available) {
        elements.dockerStatusText.textContent = 'Docker: Need Img';
        elements.dockerStatusPill.className = 'stat-pill warning';
        elements.dockerStatusPill.title = `Docker online, but image ${data.image_name} not built yet`;
      } else {
        elements.dockerStatusText.textContent = 'Docker: Offline';
        elements.dockerStatusPill.className = 'stat-pill offline';
        elements.dockerStatusPill.title = 'Docker CLI/daemon offline. Host execution strictly blocked by policy.';
      }
    } catch {
      elements.dockerStatusText.textContent = 'Docker: Offline';
      elements.dockerStatusPill.className = 'stat-pill offline';
    }
  }

  // =========================================================================
  // SESSIONS MANAGEMENT
  // =========================================================================
  async function fetchSessions() {
    try {
      const sessions = await apiGet('/api/sessions');
      state.sessions = sessions || [];
      renderSessions();

      // If we don't have an active session, activate the most recent one or start clean
      if (!state.currentSessionId && state.sessions.length > 0) {
        selectSession(state.sessions[0].id);
      } else if (state.sessions.length === 0) {
        showHeroScreen(true);
      }
    } catch (err) {
      console.warn('Could not fetch sessions:', err);
    }
  }

  function renderSessions() {
    elements.sessionCountLabel.textContent = state.sessions.length;

    if (state.sessions.length === 0) {
      elements.sessionList.innerHTML = '<div class="empty-sessions">No sessions yet</div>';
      return;
    }

    elements.sessionList.innerHTML = state.sessions
      .map(s => {
        const isActive = s.id === state.currentSessionId;
        const timeAgo = formatTimestamp(s.last_msg_at || s.created_at);
        return `
          <div class="session-item ${isActive ? 'active' : ''}" data-id="${s.id}">
            <div class="session-info">
              <span class="session-title" title="${escapeHtml(s.title || 'Untitled Session')}">
                ${escapeHtml(s.title || 'Untitled Session')}
              </span>
              <span class="session-meta">${timeAgo}</span>
            </div>
          </div>
        `;
      })
      .join('');

    // Attach click handlers
    elements.sessionList.querySelectorAll('.session-item').forEach(el => {
      el.addEventListener('click', () => {
        const sid = el.getAttribute('data-id');
        selectSession(sid);
      });
    });
  }

  async function selectSession(sessionId) {
    if (state.isGenerating) return;
    state.currentSessionId = sessionId;
    renderSessions();

    try {
      const messages = await apiGet(`/api/sessions/${sessionId}/messages`);
      state.messages = messages || [];
      renderMessages();
    } catch (err) {
      console.error('Failed to load session messages:', err);
      showToast('Failed to load conversation history', 'error');
    }
  }

  function startNewSession() {
    if (state.isGenerating) return;
    state.currentSessionId = null;
    state.messages = [];
    renderSessions();
    renderMessages();
    elements.chatInput.value = '';
    elements.chatInput.focus();
    showToast('New session started', 'info');
  }

  // =========================================================================
  // MESSAGE RENDERING
  // =========================================================================
  function renderMessages() {
    if (!state.messages || state.messages.length === 0) {
      showHeroScreen(true);
      elements.messageFeed.innerHTML = '';
      return;
    }

    showHeroScreen(false);

    elements.messageFeed.innerHTML = state.messages
      .map(msg => createMessageHtml(msg))
      .join('');

    // Attach code copy buttons
    elements.messageFeed.querySelectorAll('.copy-code-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const wrapper = e.target.closest('.code-block-wrapper');
        const code = wrapper ? wrapper.querySelector('code').innerText : '';
        if (code) {
          navigator.clipboard.writeText(code).then(() => {
            const originalText = btn.innerHTML;
            btn.innerHTML = '✓ Copied!';
            setTimeout(() => { btn.innerHTML = originalText; }, 2000);
          });
        }
      });
    });

    scrollToBottom();
  }

  function showHeroScreen(visible) {
    elements.heroScreen.style.display = visible ? 'block' : 'none';
    elements.messageFeed.style.display = visible ? 'none' : 'flex';
  }

  function createMessageHtml(msg) {
    const isUser = msg.role === 'user';
    const timeFormatted = formatTimestamp(msg.timestamp);

    if (isUser) {
      return `
        <div class="message-row user-row">
          <div class="message-bubble user-bubble">
            <div class="user-content">${escapeHtml(msg.content)}</div>
          </div>
          <div class="message-avatar" title="User">
            <span>👤</span>
          </div>
        </div>
      `;
    }

    // Assistant / Agent response
    const agentName = getAgentDisplayName(msg.agent_used);
    const parsedMarkdown = renderMarkdown(msg.content);

    return `
      <div class="message-row assistant-row">
        <div class="message-avatar" title="SciChat Assistant">
          <span>🤖</span>
        </div>
        <div class="message-bubble assistant-bubble">
          <div class="assistant-header">
            <span class="agent-tag">
              <span class="agent-icon">${getAgentIcon(msg.agent_used)}</span>
              ${escapeHtml(agentName)}
            </span>
            <span class="message-time">${timeFormatted}</span>
          </div>
          <div class="markdown-body">
            ${parsedMarkdown}
          </div>
        </div>
      </div>
    `;
  }

  function getAgentDisplayName(agentCode) {
    switch (agentCode) {
      case 'baseline_zero_shot': return 'Baseline Agent (Zero-Shot, No RAG)';
      case 'dft': return 'DFT Agent (Quantum ESPRESSO)';
      case 'training': return 'DeePMD Training Agent';
      case 'md': return 'LAMMPS MD Agent';
      case 'analysis': return 'Trajectory Analysis Agent';
      case 'qa': return 'Scientific Q&A Agent';
      case 'qa_rag':
      default: return 'Scientific RAG Orchestrator';
    }
  }

  function getAgentIcon(agentCode) {
    switch (agentCode) {
      case 'baseline_zero_shot': return '🎯';
      case 'dft': return '⚛️';
      case 'training': return '🏋️';
      case 'md': return '🔬';
      case 'analysis': return '📊';
      default: return '⚡';
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    });
  }

  // =========================================================================
  // SEND MESSAGE HANDLER
  // =========================================================================
  async function handleSendMessage() {
    const text = elements.chatInput.value.trim();
    if (!text || state.isGenerating) return;

    // Reset input
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    state.isGenerating = true;
    updateInputState();

    // Optimistically push user message to UI
    const tempUserMsg = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };
    state.messages.push(tempUserMsg);
    renderMessages();

    // Show typing indicator
    elements.typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
      const payload = {
        session_id: state.currentSessionId || undefined,
        message: text,
        mode: state.chatMode,
      };

      const response = await apiPost('/api/chat', payload);

      // Update active session ID if this was a new session
      if (response.session_id) {
        state.currentSessionId = response.session_id;
      }

      // Add assistant reply
      const assistantMsg = {
        role: 'assistant',
        content: response.reply,
        agent_used: response.agent_used || 'qa_rag',
        timestamp: new Date().toISOString(),
      };
      state.messages.push(assistantMsg);

      // Refresh sessions in background to update title
      fetchSessions();
      // Also refresh jobs if this was an orchestration action
      fetchJobs();

    } catch (err) {
      console.error('Chat error:', err);
      showToast('Failed to get agent response. Please ensure Ollama is running.', 'error');
      state.messages.push({
        role: 'assistant',
        content: `⚠️ **Connection Error**: Failed to communicate with backend or Ollama.\n\n*Error details*: \`${err.message}\`\n\nPlease ensure your Ollama instance is active and the model \`${state.status.model}\` is pulled.`,
        agent_used: 'error',
        timestamp: new Date().toISOString(),
      });
    } finally {
      elements.typingIndicator.style.display = 'none';
      state.isGenerating = false;
      updateInputState();
      renderMessages();
      elements.chatInput.focus();
    }
  }

  function updateInputState() {
    elements.sendBtn.disabled = state.isGenerating;
    elements.chatInput.disabled = state.isGenerating;
  }

  // =========================================================================
  // JOBS MONITOR
  // =========================================================================
  async function fetchJobs() {
    try {
      const jobs = await apiGet('/api/jobs');
      state.jobs = jobs || [];
      renderJobs();
    } catch (err) {
      console.warn('Could not fetch jobs:', err);
    }
  }

  function renderJobs() {
    if (!state.jobs || state.jobs.length === 0) {
      elements.jobsList.innerHTML = `
        <div class="empty-state-card">
          <div class="empty-icon">⏳</div>
          <div class="empty-title">No Active Simulations</div>
          <div class="empty-text">Jobs launched via DFT, DeePMD, or LAMMPS tools will appear here in real-time.</div>
        </div>
      `;
      elements.jobsBadgeDot.style.display = 'none';
      return;
    }

    elements.jobsBadgeDot.style.display = 'block';

    elements.jobsList.innerHTML = state.jobs
      .map(job => {
        const statusClass = (job.status || 'pending').toLowerCase();
        return `
          <div class="job-card">
            <div class="job-header">
              <span class="job-id">#${escapeHtml(String(job.id || job.slurm_id || 'LOCAL'))}</span>
              <span class="job-status-badge ${statusClass}">${escapeHtml(job.status || 'PENDING')}</span>
            </div>
            <div class="job-details">
              <strong>${escapeHtml(job.workflow_type || 'Simulation')}</strong>: ${escapeHtml(job.command || job.description || 'Running workflow task')}
            </div>
            <div class="job-meta">
              <span>${formatTimestamp(job.created_at)}</span>
              <span>${escapeHtml(job.partition || 'GPU / Compute')}</span>
            </div>
          </div>
        `;
      })
      .join('');
  }

  // =========================================================================
  // KNOWLEDGE SYNC TRIGGER
  // =========================================================================
  async function handleReindexKnowledge() {
    showToast('Re-indexing READMEs and Knowledge Graph...', 'info', 5000);
    try {
      const res = await apiPost('/api/init', {});
      showToast(`Sync complete! Ingested ${res.vectordb_chunks} chunks.`, 'success');
      state.status.vectordbChunks = res.vectordb_chunks;
      elements.chunksStat.textContent = res.vectordb_chunks;
      elements.statVectorCount.textContent = res.vectordb_chunks;
      if (res.kg_entities) {
        const totalKg = Object.values(res.kg_entities).reduce((a, b) => a + b, 0);
        elements.statKgEntities.textContent = totalKg;
      }
    } catch (err) {
      console.error('Re-index failed:', err);
      showToast('Re-indexing failed: ' + err.message, 'error');
    }
  }

  // =========================================================================
  // LIGHTWEIGHT ZERO-DEPENDENCY MARKDOWN PARSER
  // =========================================================================
  function renderMarkdown(md) {
    if (!md) return '';

    let html = md;

    // 1. Fenced code blocks ```lang ... ```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang.trim() || 'code';
      const cleanCode = escapeHtml(code.trim());
      return `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span class="code-lang">${language}</span>
            <button class="copy-code-btn" type="button">📋 Copy</button>
          </div>
          <pre><code class="language-${language}">${cleanCode}</code></pre>
        </div>
      `;
    });

    // 2. Tables
    html = html.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (match) => {
      const lines = match.trim().split('\n');
      if (lines.length < 2) return match;

      let tableHtml = '<div class="markdown-table-wrapper"><table>';
      lines.forEach((line, idx) => {
        // Skip separator line |---|---|
        if (/^\|[\s\-:|]+\|$/.test(line.trim())) return;

        const cells = line.split('|').slice(1, -1);
        const tag = idx === 0 ? 'th' : 'td';
        tableHtml += '<tr>';
        cells.forEach(cell => {
          tableHtml += `<${tag}>${formatInlineMarkdown(cell.trim())}</${tag}>`;
        });
        tableHtml += '</tr>';
      });
      tableHtml += '</table></div>';
      return tableHtml;
    });

    // 3. Blockquotes
    html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

    // 4. Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // 5. Unordered Lists
    html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // 6. Inline Formatting (bold, italic, inline code)
    html = formatInlineMarkdown(html);

    // 7. Paragraphs: split by double newline
    const paragraphs = html.split(/\n\n+/);
    html = paragraphs
      .map(p => {
        const trimmed = p.trim();
        if (!trimmed) return '';
        // Don't wrap tags that are already blocks
        if (/^<(div|ul|ol|table|blockquote|h1|h2|h3|pre)/i.test(trimmed)) {
          return trimmed;
        }
        return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    return html;
  }

  function formatInlineMarkdown(text) {
    if (!text) return '';
    return text
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Bold
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // Markdown images: ![alt](url) -> Interactive Plot Card
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
        const safeAlt = escapeHtml(alt || 'Generated Graphical Representation');
        return `<div class="chat-plot-card">` +
          `<div class="chat-plot-header">` +
            `<span class="chat-plot-title">📊 ${safeAlt}</span>` +
            `<button class="open-plot-btn" type="button" onclick="window.open('${src}', '_blank')">🔍 Expand</button>` +
          `</div>` +
          `<img class="chat-plot-image" src="${src}" alt="${safeAlt}" loading="lazy" onclick="window.open(this.src, '_blank')" />` +
        `</div>`;
      })
      // Markdown links [text](url)
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
      const date = new Date(isoStr);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }

  // =========================================================================
  // EVENT LISTENERS INITIALIZATION
  // =========================================================================
  function initEventListeners() {
    // Send button click
    elements.sendBtn.addEventListener('click', handleSendMessage);

    // Chat input Enter key handling
    elements.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    // Auto-resize textarea as user types
    elements.chatInput.addEventListener('input', () => {
      elements.chatInput.style.height = 'auto';
      elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 140) + 'px';
    });

    // Clear / New chat
    elements.clearChatBtn.addEventListener('click', startNewSession);
    elements.newChatBtn.addEventListener('click', startNewSession);

    // Toggle Sidebar
    elements.toggleSidebarBtn.addEventListener('click', () => {
      elements.sidebar.classList.toggle('collapsed');
    });

    // Toggle Inspector Drawer
    elements.toggleDrawerBtn.addEventListener('click', () => {
      elements.inspectorDrawer.classList.toggle('collapsed');
    });
    elements.closeDrawerBtn.addEventListener('click', () => {
      elements.inspectorDrawer.classList.add('collapsed');
    });

    // Drawer Tabs switching
    elements.drawerTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        elements.drawerTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabKey = tab.getAttribute('data-tab');
        Object.entries(elements.drawerPanes).forEach(([k, pane]) => {
          if (pane) pane.classList.toggle('active', k === tabKey);
        });
      });
    });

    // Refresh Jobs button
    elements.refreshJobsBtn.addEventListener('click', () => {
      fetchJobs();
      showToast('Refreshed simulation jobs', 'info', 1500);
    });

    // Sync Knowledge buttons
    elements.reindexBtn.addEventListener('click', handleReindexKnowledge);
    elements.triggerReindexBtn.addEventListener('click', handleReindexKnowledge);

    // Quick Starter Prompt cards
    document.querySelectorAll('.starter-card').forEach(card => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) {
          elements.chatInput.value = prompt;
          handleSendMessage();
        }
      });
    });

    // Mode toggle buttons (Agentic RAG vs Zero-Shot Baseline)
    document.querySelectorAll('.mode-toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.chatMode = btn.getAttribute('data-mode');
        const isBaseline = state.chatMode === 'baseline';
        const label = isBaseline ? 'Zero-Shot Baseline (No RAG)' : 'Agentic RAG';
        showToast(`Switched mode to ${label}`, 'info', 2000);
        const hint = document.querySelector('.hint-pill');
        if (hint) {
          hint.textContent = isBaseline ? 'Zero-Shot Baseline Active (No RAG)' : 'Agentic RAG Active';
          hint.style.color = isBaseline ? 'var(--status-pending)' : 'var(--accent-cyan)';
        }
      });
    });

    // Domain filter chips
    elements.domainFilters.querySelectorAll('.filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        elements.domainFilters.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        state.activeDomain = chip.getAttribute('data-domain');
        showToast(`Filtered for ${chip.textContent.trim()}`, 'info', 1500);
      });
    });
  }

  // =========================================================================
  // BOOTSTRAP
  // =========================================================================
  async function init() {
    initEventListeners();
    await checkBackendHealth();
    await checkDockerStatus();
    await fetchSessions();
    await fetchJobs();

    // Auto-poll health, docker status & jobs periodically (every 15s)
    setInterval(() => {
      checkBackendHealth();
      checkDockerStatus();
      fetchJobs();
    }, 15000);
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
