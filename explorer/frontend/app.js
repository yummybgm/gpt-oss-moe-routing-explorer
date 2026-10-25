(() => {
  'use strict';

  const SAE_LAYERS = [3, 7, 11, 15, 19, 23];
  const state = {
    meta: null,
    graph: null,
    nodeMap: new Map(),
    graphScreenNodes: [],
    selectedToken: null,
    top4: null,
    sampleTokens: [],
    selectedNodeId: null,
    selectedSaeLayer: 11,
    sae: null,
    view: { zoom: 1, panX: 0, panY: 0 },
    pointer: { down: false, moved: false, x: 0, y: 0 },
    hoveredNodeId: null,
    matrixCells: [],
  };

  const $ = (id) => document.getElementById(id);
  const elements = {
    serviceStatus: $('serviceStatus'), groupType: $('groupType'), groupValue: $('groupValue'),
    edgeMode: $('edgeMode'), edgeLimit: $('edgeLimit'), resetGraph: $('resetGraph'),
    graphScopeNote: $('graphScopeNote'), searchForm: $('searchForm'), tokenQuery: $('tokenQuery'),
    searchDomain: $('searchDomain'), searchClass: $('searchClass'), searchSegment: $('searchSegment'),
    primaryOnly: $('primaryOnly'), searchCount: $('searchCount'), resultRange: $('resultRange'),
    tokenResults: $('tokenResults'), selectedTokenSummary: $('selectedTokenSummary'),
    sampleTokens: $('sampleTokens'), sampleTokenCount: $('sampleTokenCount'),
    routingGraph: $('routingGraph'), graphStage: $('graphStage'), graphTooltip: $('graphTooltip'),
    graphLoading: $('graphLoading'), graphMetrics: $('graphMetrics'), routingMatrix: $('routingMatrix'),
    matrixStage: $('matrixStage'), matrixTooltip: $('matrixTooltip'), matrixEmpty: $('matrixEmpty'),
    matrixToken: $('matrixToken'), matrixSummary: $('matrixSummary'), contextDomain: $('contextDomain'),
    selectedNodeSummary: $('selectedNodeSummary'), contextGrid: $('contextGrid'),
    saeTabs: $('saeTabs'), saeMetrics: $('saeMetrics'), featureList: $('featureList'),
    saeToken: $('saeToken'), saeBoundary: $('saeBoundary'), toast: $('toast'),
  };

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function visibleToken(value) {
    const text = value == null ? '' : String(value);
    if (!text.length) return '∅';
    return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/\r/g, '↩');
  }

  function formatNumber(value, digits = 0) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return Number(value).toLocaleString('zh-CN', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function formatWeight(value) {
    if (value == null) return '—';
    return Number(value).toFixed(value >= 0.1 ? 3 : 4);
  }

  async function api(path, params = null) {
    const url = new URL(path, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') url.searchParams.set(key, String(value));
      });
    }
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
      } catch (_) { /* keep HTTP status */ }
      throw new Error(detail);
    }
    return response.json();
  }

  function setServiceStatus(text, mode = 'ready') {
    elements.serviceStatus.classList.toggle('is-ready', mode === 'ready');
    elements.serviceStatus.classList.toggle('is-error', mode === 'error');
    const dot = make('span', 'status-dot');
    dot.setAttribute('aria-hidden', 'true');
    elements.serviceStatus.replaceChildren(dot, document.createTextNode(text));
  }

  let toastTimer = null;
  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { elements.toast.hidden = true; }, 4200);
  }

  function fillSelect(select, values, labels = {}) {
    select.replaceChildren();
    values.forEach((value) => {
      const option = make('option', '', labels[value] || value);
      option.value = value;
      select.append(option);
    });
  }

  async function loadMeta() {
    state.meta = await api('/api/meta');
    const labels = {
      all_tokens: '全部 Token', primary_inference: '正式推断（排除首 Token）',
      'computer science': '计算机科学', law: '法律', math: '数学', physics: '物理', psychology: '心理学',
      code_symbol: '代码符号', content: '内容词', mojibake: '乱码', number: '数字', other: '其他',
      punctuation: '标点', whitespace: '空白', option_text: '选项文本', question: '问题', scaffold: '提示框架',
    };
    state.groupLabels = labels;
    updateGroupValues();
    [elements.searchDomain, elements.contextDomain].forEach((select) => {
      const firstLabel = select === elements.contextDomain ? '总体' : '全部';
      select.replaceChildren(new Option(firstLabel, ''));
      state.meta.filter_groups.domain.forEach((value) => select.append(new Option(labels[value] || value, value)));
    });
    state.meta.filter_groups.token_class.forEach((value) => elements.searchClass.append(new Option(labels[value] || value, value)));
    state.meta.filter_groups.prompt_segment.forEach((value) => elements.searchSegment.append(new Option(labels[value] || value, value)));
    renderSaeTabs();
  }

  function updateGroupValues() {
    const type = elements.groupType.value;
    fillSelect(elements.groupValue, state.meta.filter_groups[type], state.groupLabels || {});
  }

  async function loadGraph() {
    elements.graphLoading.hidden = false;
    try {
      const params = {
        group_type: elements.groupType.value,
        group_value: elements.groupValue.value,
        edge_mode: elements.edgeMode.value,
        max_edges: elements.edgeLimit.value,
      };
      state.graph = await api('/api/graph', params);
      state.nodeMap = new Map(state.graph.nodes.map((node) => [node.node_id, node]));
      renderGraphMetrics();
      drawGraph();
    } catch (error) {
      showToast(`图数据载入失败：${error.message}`);
      throw error;
    } finally {
      elements.graphLoading.hidden = true;
    }
  }

  function renderGraphMetrics() {
    const graph = state.graph;
    if (!graph || !graph.nodes.length) return;
    const tokenCount = graph.nodes[0].group_token_count;
    const displayed = graph.edges.length;
    const total = graph.edge_total_before_limit;
    elements.graphMetrics.replaceChildren();
    [
      `${formatNumber(tokenCount)} Tokens`,
      `${formatNumber(total)} 非零边`,
      graph.truncated ? `显示 ${formatNumber(displayed)}` : '完整边集',
    ].forEach((text) => elements.graphMetrics.append(make('span', 'metric-pill', text)));
    if (elements.edgeMode.value === 'top1') {
      const segments = tokenCount * 23;
      elements.graphScopeNote.textContent = `${formatNumber(tokenCount)} 条轨迹的 ${formatNumber(segments)} 个 Top‑1 跳转，经相同起终点精确聚合；没有 Token 抽样。`;
    } else {
      elements.graphScopeNote.textContent = `${formatNumber(tokenCount)} 个 Token 的 Top‑4 相邻层加权共现；它不是因果信息流。`;
    }
  }

  function layerColor(layer, alpha = 1) {
    const stops = [[68,57,131],[49,104,142],[33,145,140],[53,183,121],[253,231,37]];
    const p = (layer / 23) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(p));
    const t = p - i;
    const rgb = stops[i].map((v, index) => Math.round(v + (stops[i + 1][index] - v) * t));
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
  }

  function canvasSize(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const context = canvas.getContext('2d');
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width, height };
  }

  function graphPosition(node, width, height) {
    const scale = Math.min(width, height) * 0.43 * state.view.zoom;
    return {
      x: width / 2 + state.view.panX + node.x * scale,
      y: height / 2 + state.view.panY - node.y * scale,
    };
  }

  function drawGraph() {
    const { context: ctx, width, height } = canvasSize(elements.routingGraph);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#fffdf9';
    ctx.fillRect(0, 0, width, height);
    if (!state.graph) return;

    const values = state.graph.edges.map((edge) => Math.max(0, Number(edge.edge_value)));
    const maxLog = Math.log1p(Math.max(1, ...values));
    const edges = state.graph.edges;
    for (let index = edges.length - 1; index >= 0; index -= 1) {
      const edge = edges[index];
      const source = state.nodeMap.get(edge.source_node_id);
      const target = state.nodeMap.get(edge.target_node_id);
      if (!source || !target) continue;
      const a = graphPosition(source, width, height);
      const b = graphPosition(target, width, height);
      const strength = Math.log1p(Math.max(0, edge.edge_value)) / maxLog;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(55, 105, 151, ${0.018 + strength * 0.19})`;
      ctx.lineWidth = 0.28 + strength * 1.05;
      ctx.stroke();
    }

    if (state.selectedToken) {
      const path = state.selectedToken.trajectory
        .map((item) => state.nodeMap.get(item.node_id))
        .filter(Boolean)
        .map((node) => graphPosition(node, width, height));
      if (path.length) {
        ctx.beginPath();
        ctx.moveTo(path[0].x, path[0].y);
        path.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
        ctx.strokeStyle = 'rgba(214, 106, 50, 0.94)';
        ctx.lineWidth = 2.2;
        ctx.stroke();
      }
    }

    const maxRate = Math.max(...state.graph.nodes.map((node) => Number(node.top1_rate) || 0), 0.0001);
    state.graphScreenNodes = state.graph.nodes.map((node) => {
      const point = graphPosition(node, width, height);
      const radius = 1.65 + 3.1 * Math.sqrt((Number(node.top1_rate) || 0) / maxRate);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = layerColor(node.layer, 0.92);
      ctx.fill();
      return { ...point, radius: Math.max(6, radius + 3), node };
    });

    if (state.selectedToken) {
      state.selectedToken.trajectory.forEach((item) => {
        const node = state.nodeMap.get(item.node_id);
        if (!node) return;
        const point = graphPosition(node, width, height);
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4.7, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(214,106,50,0.95)';
        ctx.lineWidth = 1.7;
        ctx.stroke();
      });
    }

    [state.selectedNodeId, state.hoveredNodeId].forEach((nodeId, index) => {
      if (nodeId == null) return;
      const node = state.nodeMap.get(nodeId);
      if (!node) return;
      const point = graphPosition(node, width, height);
      ctx.beginPath();
      ctx.arc(point.x, point.y, index === 0 ? 7.6 : 6.2, 0, Math.PI * 2);
      ctx.strokeStyle = index === 0 ? '#d66a32' : '#173d62';
      ctx.lineWidth = index === 0 ? 2.5 : 1.5;
      ctx.stroke();
    });
  }

  function hitGraphNode(x, y) {
    let nearest = null;
    let distance = Infinity;
    state.graphScreenNodes.forEach((item) => {
      const d = Math.hypot(item.x - x, item.y - y);
      if (d <= item.radius && d < distance) { nearest = item.node; distance = d; }
    });
    return nearest;
  }

  function graphPointerPosition(event) {
    const rect = elements.routingGraph.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function showGraphTooltip(node, x, y) {
    if (!node) { elements.graphTooltip.hidden = true; return; }
    elements.graphTooltip.textContent = `Layer ${node.layer} · Expert ${node.expert_id} · Top‑1 ${formatNumber(node.top1_count)} · rate ${Number(node.top1_rate).toFixed(4)}`;
    elements.graphTooltip.hidden = false;
    const stage = elements.graphStage.getBoundingClientRect();
    elements.graphTooltip.style.left = `${Math.min(stage.width - 225, Math.max(8, x + 12))}px`;
    elements.graphTooltip.style.top = `${Math.min(stage.height - 54, Math.max(8, y + 12))}px`;
  }

  function resetGraphView() {
    state.view = { zoom: 1, panX: 0, panY: 0 };
    drawGraph();
  }

  async function searchTokens() {
    const params = {
      q: elements.tokenQuery.value,
      domain: elements.searchDomain.value,
      token_class: elements.searchClass.value,
      prompt_segment: elements.searchSegment.value,
      primary_only: elements.primaryOnly.checked,
      limit: 50,
    };
    try {
      const result = await api('/api/tokens/search', params);
      elements.searchCount.textContent = formatNumber(result.total_matches);
      elements.resultRange.textContent = result.items.length ? `显示前 ${result.items.length} 条` : '无结果';
      renderTokenResults(result.items);
    } catch (error) {
      showToast(`Token 搜索失败：${error.message}`);
    }
  }

  function renderTokenResults(items) {
    elements.tokenResults.replaceChildren();
    if (!items.length) {
      elements.tokenResults.append(make('p', 'empty-state', '没有符合当前条件的 Token。'));
      return;
    }
    items.forEach((item) => {
      const button = make('button', 'token-result');
      button.type = 'button';
      button.setAttribute('role', 'listitem');
      if (state.selectedToken?.token?.token_row === item.token_row) button.classList.add('is-selected');
      button.append(make('span', 'token-result-piece', visibleToken(item.token_text)));
      const detail = make('span', 'token-result-detail', `${item.domain} · pos ${item.token_position} · ${item.token_class}`);
      button.append(detail);
      button.addEventListener('click', () => loadToken(item.token_row));
      elements.tokenResults.append(button);
    });
  }

  function renderSelectedToken() {
    const token = state.selectedToken?.token;
    elements.selectedTokenSummary.replaceChildren();
    if (!token) {
      elements.selectedTokenSummary.append(make('p', 'empty-state', '选择一个 Token 后，将在所有面板中联动高亮。'));
      return;
    }
    const main = make('div', 'selected-token-main');
    main.append(make('span', 'token-piece', visibleToken(token.token_text)));
    main.append(make('span', 'token-id', `Token #${formatNumber(token.token_row)}`));
    elements.selectedTokenSummary.append(main);
    const chips = make('div', 'token-meta');
    [token.domain, `位置 ${token.token_position}`, token.token_class, token.prompt_segment, `路径频次 ${formatNumber(token.top1_path_frequency)}`]
      .forEach((text) => chips.append(make('span', 'meta-chip', text)));
    elements.selectedTokenSummary.append(chips);
  }

  async function loadToken(tokenRow) {
    try {
      const [detail, top4] = await Promise.all([
        api(`/api/tokens/${tokenRow}`),
        api(`/api/tokens/${tokenRow}/routing-top4`),
      ]);
      state.selectedToken = detail;
      state.top4 = top4;
      renderSelectedToken();
      drawGraph();
      drawMatrix();
      elements.matrixToken.textContent = `#${formatNumber(tokenRow)} ${visibleToken(detail.token.token_text)}`;
      elements.saeToken.textContent = `#${formatNumber(tokenRow)}`;
      await Promise.all([loadSampleTokens(detail.token.sample_id), loadSae()]);
      const focus = detail.trajectory.find((item) => item.layer === state.selectedSaeLayer);
      if (focus) await selectNode(focus.node_id);
      await searchTokens();
    } catch (error) {
      showToast(`Token 载入失败：${error.message}`);
    }
  }

  async function loadSampleTokens(sampleId) {
    const result = await api(`/api/samples/${encodeURIComponent(sampleId)}/tokens`);
    state.sampleTokens = result.items;
    elements.sampleTokenCount.textContent = `${formatNumber(result.token_count)} Tokens`;
    renderSampleTokens();
  }

  function renderSampleTokens() {
    elements.sampleTokens.replaceChildren();
    const selectedPosition = state.selectedToken?.token?.token_position ?? 0;
    const start = Math.max(0, selectedPosition - 24);
    const end = Math.min(state.sampleTokens.length, selectedPosition + 25);
    state.sampleTokens.slice(start, end).forEach((token) => {
      const button = make('button', 'sample-token', visibleToken(token.token_text));
      button.type = 'button';
      button.setAttribute('aria-label', `位置 ${token.token_position}，Token ${visibleToken(token.token_text)}`);
      if (token.token_row === state.selectedToken?.token?.token_row) button.classList.add('is-selected');
      button.addEventListener('click', () => loadToken(token.token_row));
      elements.sampleTokens.append(button);
    });
    elements.sampleTokens.querySelector('.is-selected')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  function drawMatrix() {
    const { context: ctx, width, height } = canvasSize(elements.routingMatrix);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#fffdf9';
    ctx.fillRect(0, 0, width, height);
    state.matrixCells = [];
    if (!state.top4) { elements.matrixEmpty.hidden = false; return; }
    elements.matrixEmpty.hidden = true;
    const margin = { left: 38, right: 12, top: 14, bottom: 30 };
    const plotWidth = Math.max(1, width - margin.left - margin.right);
    const plotHeight = Math.max(1, height - margin.top - margin.bottom);
    const cellWidth = plotWidth / 24;
    const cellHeight = plotHeight / 32;
    ctx.strokeStyle = 'rgba(70,82,92,0.08)';
    ctx.lineWidth = 1;
    for (let layer = 0; layer <= 24; layer += 1) {
      const x = margin.left + layer * cellWidth;
      ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotHeight); ctx.stroke();
    }
    for (let expert = 0; expert <= 32; expert += 1) {
      const y = margin.top + expert * cellHeight;
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotWidth, y); ctx.stroke();
    }
    state.top4.layers.forEach((layerRow) => {
      layerRow.selections.forEach((selection) => {
        const x = margin.left + layerRow.layer * cellWidth;
        const y = margin.top + selection.expert_id * cellHeight;
        const alpha = 0.14 + Math.min(1, selection.routing_weight) * 0.86;
        ctx.fillStyle = `rgba(36, 93, 136, ${alpha})`;
        ctx.fillRect(x + 1, y + 1, Math.max(1, cellWidth - 2), Math.max(1, cellHeight - 2));
        if (selection.expert_rank === 1) {
          ctx.strokeStyle = '#d66a32';
          ctx.lineWidth = Math.max(1.2, Math.min(2, cellWidth * 0.12));
          ctx.strokeRect(x + 1, y + 1, Math.max(1, cellWidth - 2), Math.max(1, cellHeight - 2));
        }
        state.matrixCells.push({ x, y, width: cellWidth, height: cellHeight, layer: layerRow.layer, ...selection });
      });
    });
    if (state.selectedNodeId != null) {
      const layer = Math.floor(state.selectedNodeId / 32);
      const expert = state.selectedNodeId % 32;
      const x = margin.left + layer * cellWidth;
      const y = margin.top + expert * cellHeight;
      ctx.strokeStyle = '#17212b';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, cellWidth - 1), Math.max(1, cellHeight - 1));
    }
    ctx.fillStyle = '#65707a';
    ctx.font = '9px ui-sans-serif, sans-serif';
    ctx.textAlign = 'center';
    [0,4,8,12,16,20,23].forEach((layer) => ctx.fillText(String(layer), margin.left + (layer + 0.5) * cellWidth, height - 10));
    ctx.textAlign = 'right';
    [0,8,16,24,31].forEach((expert) => ctx.fillText(String(expert), margin.left - 6, margin.top + (expert + 0.7) * cellHeight));
    ctx.save();
    ctx.translate(11, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Expert ID', 0, 0);
    ctx.restore();
    ctx.textAlign = 'center';
    ctx.fillText('Layer', margin.left + plotWidth / 2, height - 1);
    const top1 = state.top4.layers.map((layer) => layer.selections[0].expert_id).join(' → ');
    elements.matrixSummary.textContent = `Top‑1 路径：${top1}。色深表示该层归一化 Top‑4 权重。`;
  }

  function matrixPointerPosition(event) {
    const rect = elements.routingMatrix.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function hitMatrixCell(x, y) {
    return state.matrixCells.find((cell) => x >= cell.x && x <= cell.x + cell.width && y >= cell.y && y <= cell.y + cell.height) || null;
  }

  async function selectNode(nodeId) {
    state.selectedNodeId = nodeId;
    drawGraph();
    drawMatrix();
    await loadContexts();
  }

  async function loadContexts() {
    elements.contextGrid.replaceChildren();
    if (state.selectedNodeId == null) {
      elements.selectedNodeSummary.textContent = '点击主图节点或矩阵单元格以查看对应专家。';
      return;
    }
    const layer = Math.floor(state.selectedNodeId / 32);
    const expert = state.selectedNodeId % 32;
    elements.selectedNodeSummary.textContent = `Layer ${layer} · Expert ${expert} · 节点 ${state.selectedNodeId}`;
    try {
      const result = await api(`/api/nodes/${state.selectedNodeId}/contexts`, {
        domain: elements.contextDomain.value,
        limit: 12,
      });
      if (!result.items.length) {
        elements.contextGrid.append(make('p', 'empty-state', '当前证据范围没有预选上下文。'));
        return;
      }
      result.items.forEach((item) => {
        const card = make('article', 'context-card');
        const head = make('div', 'context-card-head');
        head.append(make('span', '', `#${item.display_rank} · ${item.domain}`));
        head.append(make('span', '', `w=${formatWeight(item.routing_weight)}`));
        card.append(head);
        const token = make('span', 'context-token', visibleToken(item.token_text));
        const text = make('p');
        text.append(token, document.createTextNode(`  ${item.context_text}`));
        card.append(text);
        elements.contextGrid.append(card);
      });
    } catch (error) {
      elements.contextGrid.append(make('p', 'empty-state', `上下文读取失败：${error.message}`));
    }
  }

  function renderSaeTabs() {
    elements.saeTabs.replaceChildren();
    SAE_LAYERS.forEach((layer) => {
      const button = make('button', 'sae-tab', `L${layer}`);
      button.type = 'button';
      button.classList.toggle('is-active', layer === state.selectedSaeLayer);
      button.setAttribute('aria-pressed', String(layer === state.selectedSaeLayer));
      button.addEventListener('click', async () => {
        state.selectedSaeLayer = layer;
        renderSaeTabs();
        await loadSae();
        const focus = state.selectedToken?.trajectory.find((item) => item.layer === layer);
        if (focus) await selectNode(focus.node_id);
      });
      elements.saeTabs.append(button);
    });
  }

  async function loadSae() {
    if (!state.selectedToken) return;
    try {
      state.sae = await api(`/api/tokens/${state.selectedToken.token.token_row}/sae`, {
        sae_layer: state.selectedSaeLayer,
        limit: 18,
      });
      renderSae();
    } catch (error) {
      elements.featureList.replaceChildren(make('p', 'empty-state', `SAE 读取失败：${error.message}`));
    }
  }

  function renderSae() {
    const result = state.sae;
    elements.saeMetrics.replaceChildren();
    elements.featureList.replaceChildren();
    if (!result) return;
    const metrics = [
      `活跃 ${formatNumber(result.total_active_features)}`,
      `Cosine ${Number(result.metrics.cosine_similarity).toFixed(3)}`,
      `MSE ${Number(result.metrics.reconstruction_mse).toExponential(2)}`,
    ];
    metrics.forEach((text) => elements.saeMetrics.append(make('span', 'metric-pill', text)));
    if (!result.items.length) {
      elements.featureList.append(make('p', 'empty-state', '该 Token 在当前阈值下没有返回活跃特征。'));
    } else {
      const maximum = Math.max(...result.items.map((item) => Number(item.activation)), 1e-12);
      result.items.forEach((item) => {
        const row = make('div', 'feature-row');
        row.append(make('span', 'feature-id', `F${item.feature_id}`));
        const track = make('div', 'feature-bar-track');
        const bar = make('div', 'feature-bar');
        bar.style.width = `${Math.max(1, (Number(item.activation) / maximum) * 100)}%`;
        track.append(bar);
        row.append(track);
        row.append(make('span', 'feature-value', formatWeight(item.activation)));
        elements.featureList.append(row);
      });
    }
    const firstWarning = result.warnings.find((warning) => warning.startsWith('First-token'));
    elements.saeBoundary.textContent = firstWarning
      ? `⚠ ${firstWarning} 当前只显示 ID 与激活强度，不自动命名语义。`
      : '当前只显示 feature ID 与激活强度；没有验证过的全局语义时不会自动命名。';
  }

  function bindEvents() {
    elements.groupType.addEventListener('change', async () => { updateGroupValues(); await loadGraph(); });
    elements.groupValue.addEventListener('change', loadGraph);
    elements.edgeMode.addEventListener('change', loadGraph);
    elements.edgeLimit.addEventListener('change', loadGraph);
    elements.resetGraph.addEventListener('click', resetGraphView);
    elements.searchForm.addEventListener('submit', (event) => { event.preventDefault(); searchTokens(); });
    elements.contextDomain.addEventListener('change', loadContexts);

    elements.routingGraph.addEventListener('pointerdown', (event) => {
      const point = graphPointerPosition(event);
      state.pointer = { down: true, moved: false, x: point.x, y: point.y };
      elements.routingGraph.setPointerCapture(event.pointerId);
    });
    elements.routingGraph.addEventListener('pointermove', (event) => {
      const point = graphPointerPosition(event);
      if (state.pointer.down) {
        const dx = point.x - state.pointer.x;
        const dy = point.y - state.pointer.y;
        if (Math.hypot(dx, dy) > 1) state.pointer.moved = true;
        state.view.panX += dx;
        state.view.panY += dy;
        state.pointer.x = point.x;
        state.pointer.y = point.y;
        elements.graphTooltip.hidden = true;
        drawGraph();
      } else {
        const node = hitGraphNode(point.x, point.y);
        const nodeId = node?.node_id ?? null;
        if (nodeId !== state.hoveredNodeId) {
          state.hoveredNodeId = nodeId;
          drawGraph();
        }
        showGraphTooltip(node, point.x, point.y);
      }
    });
    elements.routingGraph.addEventListener('pointerup', async (event) => {
      const point = graphPointerPosition(event);
      if (state.pointer.down && !state.pointer.moved) {
        const node = hitGraphNode(point.x, point.y);
        if (node) await selectNode(node.node_id);
      }
      state.pointer.down = false;
      try { elements.routingGraph.releasePointerCapture(event.pointerId); } catch (_) { /* capture already released */ }
    });
    elements.routingGraph.addEventListener('pointerleave', () => {
      state.pointer.down = false;
      state.hoveredNodeId = null;
      elements.graphTooltip.hidden = true;
      drawGraph();
    });
    elements.routingGraph.addEventListener('dblclick', resetGraphView);
    elements.routingGraph.addEventListener('wheel', (event) => {
      event.preventDefault();
      const point = graphPointerPosition(event);
      const { width, height } = elements.routingGraph.getBoundingClientRect();
      const oldZoom = state.view.zoom;
      const newZoom = Math.min(8, Math.max(0.45, oldZoom * Math.exp(-event.deltaY * 0.0012)));
      const ratio = newZoom / oldZoom;
      const oldCenterX = width / 2 + state.view.panX;
      const oldCenterY = height / 2 + state.view.panY;
      state.view.panX = point.x - (point.x - oldCenterX) * ratio - width / 2;
      state.view.panY = point.y - (point.y - oldCenterY) * ratio - height / 2;
      state.view.zoom = newZoom;
      drawGraph();
    }, { passive: false });

    elements.routingMatrix.addEventListener('mousemove', (event) => {
      const point = matrixPointerPosition(event);
      const cell = hitMatrixCell(point.x, point.y);
      if (!cell) { elements.matrixTooltip.hidden = true; return; }
      elements.matrixTooltip.textContent = `Layer ${cell.layer} · Expert ${cell.expert_id} · Rank ${cell.expert_rank} · weight ${formatWeight(cell.routing_weight)}`;
      elements.matrixTooltip.hidden = false;
      const stage = elements.matrixStage.getBoundingClientRect();
      elements.matrixTooltip.style.left = `${Math.min(stage.width - 218, Math.max(8, point.x + 10))}px`;
      elements.matrixTooltip.style.top = `${Math.min(stage.height - 50, Math.max(8, point.y + 10))}px`;
    });
    elements.routingMatrix.addEventListener('mouseleave', () => { elements.matrixTooltip.hidden = true; });
    elements.routingMatrix.addEventListener('click', async (event) => {
      const point = matrixPointerPosition(event);
      const cell = hitMatrixCell(point.x, point.y);
      if (cell) await selectNode(cell.node_id);
    });

    const observer = new ResizeObserver(() => { drawGraph(); drawMatrix(); });
    observer.observe(elements.graphStage);
    observer.observe(elements.matrixStage);
  }

  async function init() {
    bindEvents();
    if (window.location.protocol === 'file:') {
      setServiceStatus('打开方式不正确 · 请使用一键启动入口', 'error');
      elements.graphLoading.hidden = false;
      elements.graphLoading.textContent = '请关闭此页面，双击项目根目录中的“打开GPT-OSS交互图.cmd”。';
      elements.graphScopeNote.textContent = '该页面需要只读数据服务；直接双击 index.html 无法加载实验缓存。';
      showToast('请使用“打开GPT-OSS交互图.cmd”启动数据服务和页面。');
      return;
    }
    try {
      await loadMeta();
      await Promise.all([loadGraph(), searchTokens()]);
      await loadToken(1);
      setServiceStatus('数据服务已连接 · 全量模式', 'ready');
    } catch (error) {
      setServiceStatus('数据服务连接失败', 'error');
      showToast(`初始化失败：${error.message}`);
    }
  }

  init();
})();
