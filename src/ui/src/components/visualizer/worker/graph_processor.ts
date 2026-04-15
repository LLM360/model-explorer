/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import {
  DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
  TENSOR_TAG_METADATA_KEY,
  TENSOR_VALUES_KEY,
} from '../common/consts';
import {Graph, GraphNode} from '../common/input_graph';
import {
  EdgeColorRole,
  EdgeStyleData,
  GroupNode,
  ModelGraph,
  ModelNode,
  NodeType,
  OpNode,
} from '../common/model_graph';
import {
  KeyValuePairs,
  MetadataItem,
  NodeAttributeList,
  NodeAttributePairs,
  NodeAttributeValue,
  NodeDataProviderRunData,
  OutgoingEdge,
  ShowOnNodeItemData,
} from '../common/types';
import {
  findCommonNamespace,
  getNextLevelNsPart,
  isGroupNode,
  isOpNode,
  splitNamespace,
  unEscapeString,
} from '../common/utils';
import {VisualizerConfig} from '../common/visualizer_config';
import {ProcessingLabel} from '../common/worker_events';

import {getLayoutGraph} from './graph_layout';
import {updateProcessingProgress} from './utils';

const CONST_VALUE_REGEX = /dense<([^>]*)>/;

/**
 * A class that processes a given `Graph` into a `ModelGraph`.
 */
export class GraphProcessor {
  private readonly activeGraph: Graph;
  private readonly nodeLabelsToHide: Set<string>;

  constructor(
    private readonly paneId: string,
    private readonly graph: Graph,
    private readonly config?: VisualizerConfig,
    private readonly showOnNodeItemTypes: Record<
      string,
      ShowOnNodeItemData
    > = {},
    private readonly nodeDataProviderRuns: Record<
      string,
      NodeDataProviderRunData
    > = {},
    private readonly groupNodeChildrenCountThreshold = DEFAULT_GROUP_NODE_CHILDREN_COUNT_THRESHOLD,
    private readonly testMode = false,
    private readonly flattenLayers = false,
    private readonly architectureMode = false,
    private readonly hideShapeNodes = false,
    private readonly keepLayersWithASingleChild = false,
  ) {
    this.activeGraph = this.architectureMode
      ? buildArchitectureGraph(this.graph)
      : this.graph;
    this.nodeLabelsToHide = new Set<string>(
      (this.activeGraph.nodeLabelsToHide ?? this.config?.nodeLabelsToHide ?? []).map(
        (label) => label.toLowerCase(),
      ),
    );
  }

  process(): ModelGraph {
    const modelGraph = this.createEmptyModelGraph();

    this.processNodes(modelGraph);
    this.processEdgeRelationships(modelGraph);
    this.classifyShapeNodes(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_NODES_AND_EDGES,
    );

    this.processNamespaceRelationships(modelGraph);
    this.pruneEmptyGroupNodes(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_LAYER_NAMESPACES,
    );

    this.generateLayoutGraphConnections(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.PROCESSING_LAYOUT_DATA,
    );

    this.splitLargeGroupNodes(modelGraph);
    updateProcessingProgress(
      this.paneId,
      ProcessingLabel.SPLITTING_LARGE_LAYERS,
    );

    this.populateDescendantsAndCounts(modelGraph);

    return modelGraph;
  }

  /**
   * Scans nodes in `Graph` and creates the corresponding `OpNode` and
   * `GroupNode` in the `ModelGraph` (see model_graph.ts for more details).
   */
  processNodes(modelGraph: ModelGraph) {
    const seenNamespaces = new Set<string>();
    for (const graphNode of this.activeGraph.nodes) {
      // Add an `OpNode` to the model graph for each node in the input graph.
      //
      // If namespace is a ";" separated string, use the last component as the
      // actual namespace.
      const namespace = graphNode.namespace;
      const parts = namespace.split(';').filter((part) => part !== '');
      if (parts.length > 1) {
        graphNode.namespace = parts[parts.length - 1];
      }
      const opNode: OpNode = {
        nodeType: NodeType.OP_NODE,
        id: graphNode.id,
        namespace: this.flattenLayers ? '' : graphNode.namespace,
        savedNamespace: graphNode.namespace,
        fullNamespace: graphNode.namespace,
        label: graphNode.label,
        level: splitNamespace(graphNode.namespace).length,
      };
      if (graphNode.subgraphIds && graphNode.subgraphIds.length > 0) {
        opNode.subgraphIds = graphNode.subgraphIds;
      }
      if (this.nodeLabelsToHide.has(graphNode.label.toLowerCase())) {
        opNode.hideInLayout = true;
      }
      if (this.config?.nodeAttrsToHide) {
        const nodeAttrsWithBasicInfo: NodeAttributeList = [];
        if (graphNode.attrs != null) {
          nodeAttrsWithBasicInfo.push(...graphNode.attrs);
        }
        nodeAttrsWithBasicInfo.push({
          key: 'id',
          value: graphNode.id,
        });
        nodeAttrsWithBasicInfo.push({
          key: 'name',
          value: graphNode.label,
        });
        nodeAttrsWithBasicInfo.push({
          key: 'namespace',
          value: graphNode.namespace,
        });
        for (const [attrKey, attrValueRegex] of Object.entries(
          this.config.nodeAttrsToHide,
        )) {
          const attrValue = nodeAttrsWithBasicInfo.find(
            (attr) => attr.key === attrKey,
          )?.value;
          if (
            attrValue &&
            typeof attrValue === 'string' &&
            attrValue.match(attrValueRegex)
          ) {
            opNode.hideInLayout = true;
            break;
          }
        }
      }
      if (graphNode.attrs) {
        const attrs: NodeAttributePairs = {};
        for (const attr of graphNode.attrs) {
          attrs[attr.key] = this.processAttrValue(attr.key, attr.value);
        }
        opNode.attrs = attrs;
      }
      if (graphNode.inputsMetadata) {
        opNode.inputsMetadata = this.processMetadataList(
          graphNode.inputsMetadata,
        );
      }
      if (graphNode.outputsMetadata) {
        opNode.outputsMetadata = this.processMetadataList(
          graphNode.outputsMetadata,
        );
      }
      if (graphNode.style) {
        opNode.style = graphNode.style;
      }
      if (graphNode.config) {
        opNode.config = graphNode.config;
      }
      modelGraph.nodes.push(opNode);
      modelGraph.nodesById[opNode.id] = opNode;

      // Add group nodes for all ancestor namespaces from this op node.
      //
      // For example, if an op node's namespace is a/b/c, then add the following
      // group nodes.
      //
      // - namespace: a/b, label: c.
      // - namespace: a, label: b.
      // - namespace: <empty>, label a.
      if (!opNode.hideInLayout && !this.flattenLayers) {
        const ancestorNamespaces = this.getAncestorNamespaces(opNode.namespace);
        for (const ns of ancestorNamespaces) {
          if (seenNamespaces.has(ns)) {
            continue;
          }
          seenNamespaces.add(ns);

          const components = splitNamespace(ns);
          // Use the last component of the namespace as its display label.
          const label = unEscapeString(components.splice(-1)[0]);
          // Group node's namespace doesn't contain the last component.
          const namespace = components.join('/');
          const groupNode: GroupNode = {
            nodeType: NodeType.GROUP_NODE,
            id: this.getGroupNodeIdFromNamespace(ns),
            namespace,
            label,
            level: components.length,
            expanded: false,
          };
          modelGraph.nodes.push(groupNode);
          modelGraph.nodesById[groupNode.id] = groupNode;
        }
      }
    }
  }

  private classifyShapeNodes(modelGraph: ModelGraph) {
    if (!this.hideShapeNodes) {
      return;
    }
    // Prefer explicit tensor-tag metadata and fall back to shape-only edge
    // flow only when the node has no data edges.
    for (const node of modelGraph.nodes) {
      if (!isOpNode(node) || node.hideInLayout) {
        continue;
      }
      if (this.shouldHideShapeNode(node)) {
        node.hideInLayout = true;
      }
    }
  }

  private pruneEmptyGroupNodes(modelGraph: ModelGraph) {
    while (true) {
      let removedAny = false;
      for (const node of [...modelGraph.nodes]) {
        if (!isGroupNode(node)) {
          continue;
        }
        if ((node.nsChildrenIds || []).length > 0) {
          continue;
        }

        removedAny = true;
        const nodeIndex = modelGraph.nodes.indexOf(node);
        if (nodeIndex >= 0) {
          modelGraph.nodes.splice(nodeIndex, 1);
        }
        delete modelGraph.nodesById[node.id];

        const rootIndex = modelGraph.rootNodes.indexOf(node);
        if (rootIndex >= 0) {
          modelGraph.rootNodes.splice(rootIndex, 1);
        }

        if (node.nsParentId) {
          const parentNode = modelGraph.nodesById[node.nsParentId] as GroupNode;
          if (parentNode?.nsChildrenIds) {
            parentNode.nsChildrenIds = parentNode.nsChildrenIds.filter(
              (childId) => childId !== node.id,
            );
          }
        }

        if (modelGraph.artificialGroupNodeIds != null) {
          modelGraph.artificialGroupNodeIds =
            modelGraph.artificialGroupNodeIds.filter((id) => id !== node.id);
        }
      }

      if (!removedAny) {
        break;
      }
    }
  }

  /**
   * Sets edges in the given model graph based on the edges in the input graph.
   */
  processEdgeRelationships(modelGraph: ModelGraph) {
    for (const graphNode of this.activeGraph.nodes) {
      const node = modelGraph.nodesById[graphNode.id] as OpNode;
      if (!node) {
        continue;
      }

      // From the graph node's incoming edges, populate the incoming and
      // outgoing edges for the corresponding node in the model graph.
      for (const incomingEdge of graphNode.incomingEdges || []) {
        const sourceNodeId = incomingEdge.sourceNodeId;
        const sourceNode = modelGraph.nodesById[sourceNodeId] as OpNode;
        if (!sourceNode) {
          continue;
        }

        // Incoming edges.
        if (node.incomingEdges == null) {
          node.incomingEdges = [];
        }
        if (
          node.incomingEdges.find(
            (edge) =>
              edge.sourceNodeId === sourceNodeId &&
              edge.sourceNodeOutputId === incomingEdge.sourceNodeOutputId &&
              edge.targetNodeInputId === incomingEdge.targetNodeInputId,
          ) == null
        ) {
          node.incomingEdges.push({...incomingEdge});
        }

        // Outgoing edges.
        if (sourceNode.outgoingEdges == null) {
          sourceNode.outgoingEdges = [];
        }
        if (
          sourceNode.outgoingEdges.find(
            (edge) =>
              edge.targetNodeId === node.id &&
              edge.sourceNodeOutputId === incomingEdge.sourceNodeOutputId &&
              edge.targetNodeInputId === incomingEdge.targetNodeInputId,
          ) == null
        ) {
          sourceNode.outgoingEdges.push({
            targetNodeId: node.id,
            sourceNodeOutputId: incomingEdge.sourceNodeOutputId,
            targetNodeInputId: incomingEdge.targetNodeInputId,
            metadata: incomingEdge.metadata,
          });
        }
      }
    }
  }

  /**
   * Sets namespace relationships in model graph based on the hierarchy data
   * stored in input node's `namespace`.
   */
  processNamespaceRelationships(modelGraph: ModelGraph) {
    for (const node of modelGraph.nodes) {
      if (isOpNode(node) && node.hideInLayout) {
        continue;
      }

      const ns = node.namespace;

      // Root node.
      if (ns === '') {
        modelGraph.rootNodes.push(node);
        continue;
      }

      // Set namespace parent.
      const parentNodeId = this.getGroupNodeIdFromNamespace(ns);
      const parentGroupNode = modelGraph.nodesById[parentNodeId] as GroupNode;
      if (parentGroupNode) {
        node.nsParentId = parentGroupNode.id;
      } else {
        console.warn(
          `Failed to find the NS parent of node "${node.id}": "${parentNodeId}"`,
        );
      }

      // Set namespace children.
      if (parentGroupNode) {
        if (parentGroupNode.nsChildrenIds == null) {
          parentGroupNode.nsChildrenIds = [];
        }
        if (!parentGroupNode.nsChildrenIds.includes(node.id)) {
          parentGroupNode.nsChildrenIds.push(node.id);
          if (isOpNode(node) && node.config?.pinToGroupTop) {
            if (parentGroupNode.pinToTopOpNodes == null) {
              parentGroupNode.pinToTopOpNodes = [];
            }
            parentGroupNode.pinToTopOpNodes.push(node);
          }
          if (isOpNode(node) && node.config?.pinToGroupBottom) {
            if (parentGroupNode.pinToBottomOpNodes == null) {
              parentGroupNode.pinToBottomOpNodes = [];
            }
            parentGroupNode.pinToBottomOpNodes.push(node);
          }
        }
      }
    }

    // Find group nodes that only have one single op node as its child. For
    // these nodes, remove the group node and move the child op node up a level
    // from its namespace.
    //
    // Repeatedly do this until no such nodes are found.
    if (!this.keepLayersWithASingleChild) {
      while (true) {
        let numNodeProcessed = 0;
        for (const node of modelGraph.nodes) {
          if (!isGroupNode(node)) {
            continue;
          }
          if (node.nsChildrenIds != null && node.nsChildrenIds.length === 1) {
            const opNode = modelGraph.nodesById[node.nsChildrenIds[0]];
            if (isOpNode(opNode)) {
              numNodeProcessed++;
              // Delete group node.
              const index = modelGraph.nodes.indexOf(node);
              if (index >= 0) {
                modelGraph.nodes.splice(index, 1);
              }
              delete modelGraph.nodesById[node.id];

              // Move op node up one level in namespace.
              const ns = opNode.namespace;
              const parts = splitNamespace(ns);
              parts.pop();
              opNode.namespace = parts.join('/');
              opNode.savedNamespace = opNode.namespace;
              opNode.level = parts.length;
              opNode.nsParentId = node.nsParentId;

              // Update root node if necessary.
              const indexInRootNodes = modelGraph.rootNodes.indexOf(node);
              if (indexInRootNodes >= 0) {
                modelGraph.rootNodes.splice(indexInRootNodes, 1);
                modelGraph.rootNodes.push(opNode);
              }

              // Remove this node from its NS parent node's nsChildrenIds, and add
              // the op node to it.
              if (node.nsParentId) {
                const nsParent = modelGraph.nodesById[
                  node.nsParentId
                ] as GroupNode;
                const index = nsParent.nsChildrenIds!.indexOf(node.id);
                nsParent.nsChildrenIds!.splice(index, 1);
                nsParent.nsChildrenIds!.push(opNode.id);
              }
            }
          }
        }
        if (numNodeProcessed === 0) {
          break;
        }
      }
    }
  }

  /**
   * Generates layout graph connections for the given model graph.
   */
  generateLayoutGraphConnections(modelGraph: ModelGraph) {
    modelGraph.layoutGraphEdges = {};

    // Find all op nodes that don't have incoming edges.
    let seedOpNodes: OpNode[] = [];
    const allNonHiddenOpNodes: OpNode[] = [];
    for (const node of modelGraph.nodes) {
      if (!isOpNode(node) || node.hideInLayout) {
        continue;
      }
      allNonHiddenOpNodes.push(node);
      const filteredIncomingEdges = (node.incomingEdges || []).filter(
        (edge) =>
          !(modelGraph.nodesById[edge.sourceNodeId] as OpNode).hideInLayout,
      );
      if (filteredIncomingEdges.length === 0) {
        seedOpNodes.push(node);
      }
    }

    // If seedOpNodes is empty, it means all the nodes in the graph have
    // incoming edges. This indicates that the graph must contain at least one
    // full cycle without any "root" nodes. For example, the graph might have
    // one circle, or two disjoint circles, etc.
    //
    // Instead of picking one node from each of these disjointed cycles (which
    // might be expensive to calculate), we will just use all the nodes in the
    // graph as the seed nodes. The DFS procedure below will handle the dedup
    // logic correctly.
    if (seedOpNodes.length === 0 && allNonHiddenOpNodes.length > 0) {
      seedOpNodes = allNonHiddenOpNodes;
    }

    // Do a BFS from seedOpNodes.
    const queue: OpNode[] = [...seedOpNodes];
    const seenNodeIds = new Set<string>();
    while (queue.length > 0) {
      const curNode = queue.shift();
      if (curNode == null || curNode.hideInLayout) {
        continue;
      }
      if (seenNodeIds.has(curNode.id)) {
        continue;
      }
      seenNodeIds.add(curNode.id);

      // For each edge going from curNode (A), find the common namespace of
      // curNode and edge's target node (B), and mark the connection between the
      // top-level node that contains A and B within the common namespace.
      //
      // For example, op node X's namespae is a/b/c, op node Y's namespace
      // is a/b/d, and X has an edge to Y. X and Y's common namespace is a/b.
      // So we mark a/b/c and a/b/d to be connected.
      const outgoingEdges = curNode.outgoingEdges || [];
      for (const edge of outgoingEdges) {
        const targetNode = modelGraph.nodesById[edge.targetNodeId] as OpNode;
        if (targetNode.hideInLayout) {
          continue;
        }
        const commonNs = findCommonNamespace(
          curNode.namespace,
          targetNode.namespace,
        );
        const sourceNodeNextLevelNsPart = getNextLevelNsPart(
          commonNs,
          curNode.namespace,
        );
        const connectionFromNodeId =
          sourceNodeNextLevelNsPart === ''
            ? curNode.id
            : `${commonNs}${
                commonNs === '' ? '' : '/'
              }${sourceNodeNextLevelNsPart}___group___`;
        const targetNodeNextLevelNsPart = getNextLevelNsPart(
          commonNs,
          targetNode.namespace,
        );
        const connectionToNodeId =
          targetNodeNextLevelNsPart === ''
            ? targetNode.id
            : `${commonNs}${
                commonNs === '' ? '' : '/'
              }${targetNodeNextLevelNsPart}___group___`;

        const commonNsGroupId = commonNs === '' ? '' : `${commonNs}___group___`;
        if (modelGraph.layoutGraphEdges[commonNsGroupId] == null) {
          modelGraph.layoutGraphEdges[commonNsGroupId] = {};
        }
        if (
          modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId] ==
          null
        ) {
          modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId] =
            {};
        }
        const existingEdgeStyle =
          modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId][
            connectionToNodeId
          ];
        modelGraph.layoutGraphEdges[commonNsGroupId][connectionFromNodeId][
          connectionToNodeId
        ] = mergeEdgeStyleData(
          existingEdgeStyle,
          getOutgoingEdgeStyleData(edge),
        );
      }

      for (const edge of outgoingEdges) {
        const targetNode = modelGraph.nodesById[edge.targetNodeId] as OpNode;
        queue.push(targetNode);
      }
    }
  }

  /**
   * Finds group nodes with a large number of children, and splits them into
   * different groups
   */
  splitLargeGroupNodes(modelGraph: ModelGraph) {
    // From root, do a BFS search on all group nodes.
    const queue: Array<GroupNode | undefined> = [undefined];
    let hasLargeGroupNodes = false;
    while (queue.length > 0) {
      const curGroupNode = queue.shift();
      let children: ModelNode[] =
        curGroupNode == null
          ? modelGraph.rootNodes
          : (curGroupNode.nsChildrenIds || []).map(
              (id) => modelGraph.nodesById[id],
            );

      // Split the group node if its child count is over the threshold.
      if (children.length > this.groupNodeChildrenCountThreshold) {
        hasLargeGroupNodes = true;
        const layoutGraph = getLayoutGraph(
          curGroupNode?.id || '',
          children,
          modelGraph,
          this.showOnNodeItemTypes,
          this.nodeDataProviderRuns,
          undefined,
          this.testMode,
          // Use fake node size.
          true,
          this.config,
        );

        // Find root nodes of the layout graph.
        const rootNodes: ModelNode[] = [];
        for (const nodeId of Object.keys(layoutGraph.nodes)) {
          if (layoutGraph.incomingEdges[nodeId] == null) {
            rootNodes.push(modelGraph.nodesById[nodeId]);
          }
        }

        // Do a DFS from the layout graph root nodes. Create a new group
        // whenever the node counts reaches the threshold.
        const groups: ModelNode[][] = [];
        let curGroup: ModelNode[] = [];
        const visitedNodeIds = new Set<string>();
        const visit = (curNodeId: string) => {
          if (visitedNodeIds.has(curNodeId)) {
            return;
          }
          visitedNodeIds.add(curNodeId);
          const node = modelGraph.nodesById[curNodeId];
          curGroup.push(node);
          if (curGroup.length === this.groupNodeChildrenCountThreshold) {
            groups.push(curGroup);
            curGroup = [];
          }
          for (const childId of layoutGraph.outgoingEdges[node.id] || []) {
            visit(childId);
          }
        };
        for (const rootNode of rootNodes) {
          visit(rootNode.id);
        }
        if (
          curGroup.length < this.groupNodeChildrenCountThreshold &&
          curGroup.length > 0
        ) {
          groups.push(curGroup);
        }

        // Create a new group node for each group.
        const newGroupNodes: GroupNode[] = [];
        for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
          const nodes = groups[groupIndex];
          const newGroupNodeNamespace =
            curGroupNode == null
              ? ''
              : `${curGroupNode.namespace}/${curGroupNode.label}`;
          const newGroupNodeLabel = `section_${groupIndex + 1}_of_${
            groups.length
          }`;
          const newGroupNodeId =
            curGroupNode == null
              ? `${newGroupNodeLabel}___group___`
              : `${newGroupNodeNamespace}/${newGroupNodeLabel}___group___`;
          const newGroupNode: GroupNode = {
            nodeType: NodeType.GROUP_NODE,
            id: newGroupNodeId,
            label: newGroupNodeLabel,
            namespace: newGroupNodeNamespace,
            level: splitNamespace(newGroupNodeNamespace).length,
            nsParentId: curGroupNode?.id,
            nsChildrenIds: nodes.map((node) => node.id),
            expanded: false,
            sectionContainer: true,
          };
          newGroupNodes.push(newGroupNode);

          // Add the new group node to the model graph.
          modelGraph.nodes.push(newGroupNode);
          modelGraph.nodesById[newGroupNode.id] = newGroupNode;
          if (modelGraph.artificialGroupNodeIds == null) {
            modelGraph.artificialGroupNodeIds = [];
          }
          modelGraph.artificialGroupNodeIds.push(newGroupNode.id);

          // Update the ns parent for all nodes in the new group.
          for (const node of nodes) {
            node.nsParentId = newGroupNode.id;
          }

          // Update the namespace of all nodes and their desendents in the new
          // group.
          const newNamespacePart = newGroupNodeId.replace('___group___', '');
          const updateNamespace = (node: ModelNode) => {
            const oldNamespace = node.namespace;
            if (oldNamespace === '') {
              node.namespace = newNamespacePart;
            } else {
              if (curGroupNode == null) {
                node.namespace = `${newNamespacePart}/${node.namespace}`;
              } else {
                node.namespace = (node.nsParentId || '').replace(
                  '___group___',
                  '',
                );
              }
            }
            node.level = splitNamespace(node.namespace).length;
            if (isGroupNode(node)) {
              // Update group node id since its namespace has been changed.
              const oldNodeId = node.id;
              delete modelGraph.nodesById[node.id];
              node.id = `${node.namespace}/${node.label}___group___`;
              modelGraph.nodesById[node.id] = node;

              // Update its parent's nsChildren to use the new id.
              if (node.nsParentId) {
                const nsParent = modelGraph.nodesById[
                  node.nsParentId
                ] as GroupNode;
                const index = (nsParent.nsChildrenIds || []).indexOf(oldNodeId);
                if (index >= 0) {
                  (nsParent.nsChildrenIds || [])[index] = node.id;
                }
              }

              for (const nsChildId of node.nsChildrenIds || []) {
                const childNode = modelGraph.nodesById[nsChildId];
                if (childNode != null) {
                  // Update its children's nsParent id.
                  childNode.nsParentId = node.id;
                  // BFS.
                  updateNamespace(childNode);
                }
              }
            }
          };
          for (const node of nodes) {
            updateNamespace(node);
          }

          if (curGroupNode == null) {
            // Remove the nodes in the current new group if they are in the root
            // node list.
            for (const node of nodes) {
              const index = modelGraph.rootNodes.indexOf(node);
              if (index >= 0) {
                modelGraph.rootNodes.splice(index, 1);
              }
            }

            // Add the new group node to root node list if its namespace is
            // empty.
            if (newGroupNode.namespace === '') {
              modelGraph.rootNodes.push(newGroupNode);
            }
          }

          children = newGroupNodes;
        }

        // Update curGropNode's nsChildrenIds.
        if (curGroupNode != null) {
          curGroupNode.nsChildrenIds = newGroupNodes.map((node) => node.id);
        }
      }

      for (const child of children) {
        if (isGroupNode(child)) {
          queue.push(child);
        }
      }
    }

    if (hasLargeGroupNodes) {
      this.generateLayoutGraphConnections(modelGraph);
    }
  }

  populateDescendantsAndCounts(modelGraph: ModelGraph) {
    // For each group node, gather all its descendant nodes.
    let minOpNodeCount = Number.MAX_VALUE;
    let maxOpNodeCount = Number.NEGATIVE_INFINITY;
    for (const node of modelGraph.nodes) {
      if (isGroupNode(node)) {
        const descendants: ModelNode[] = [];
        this.gatherDescendants(modelGraph, node, descendants);
        node.descendantsNodeIds = descendants.map((node) => node.id);
        node.descendantsOpNodeIds = descendants
          .filter((node) => node.nodeType === NodeType.OP_NODE)
          .map((node) => node.id);
        const opNodeCount = (node.descendantsOpNodeIds || []).length;
        minOpNodeCount = Math.min(opNodeCount, minOpNodeCount);
        maxOpNodeCount = Math.max(opNodeCount, maxOpNodeCount);
      }
    }
    modelGraph.minDescendantOpNodeCount = minOpNodeCount;
    modelGraph.maxDescendantOpNodeCount = maxOpNodeCount;
  }

  createEmptyModelGraph(): ModelGraph {
    const modelGraph: ModelGraph = {
      id: this.activeGraph.id,
      collectionLabel: this.activeGraph.collectionLabel || '',
      nodes: [],
      nodesById: {},
      rootNodes: [],
      edgesByGroupNodeIds: {},
      layoutGraphEdges: {},
      minDescendantOpNodeCount: -1,
      maxDescendantOpNodeCount: -1,
      modelPath: this.activeGraph.modelPath,
      adapterId: this.activeGraph.adapterId,
    };
    if (this.activeGraph.groupNodeAttributes) {
      modelGraph.groupNodeAttributes = this.activeGraph.groupNodeAttributes;
    }
    if (this.activeGraph.groupNodeConfigs) {
      modelGraph.groupNodeConfigs = this.activeGraph.groupNodeConfigs;
    }
    if (this.activeGraph.layoutConfigs) {
      modelGraph.layoutConfigs = this.activeGraph.layoutConfigs;
    }

    return modelGraph;
  }

  private getAncestorNamespaces(ns: string): string[] {
    // The returned namespaces include `ns` as well.
    const components = splitNamespace(ns);
    const namespaces: string[] = [];
    while (components.length > 0) {
      namespaces.push(components.join('/'));
      components.pop();
    }
    return namespaces;
  }

  private getGroupNodeIdFromNamespace(ns: string): string {
    return `${ns}___group___`;
  }

  private gatherDescendants(
    modelGraph: ModelGraph,
    curRoot: GroupNode,
    descendants: ModelNode[],
  ) {
    for (const childId of curRoot.nsChildrenIds || []) {
      const child = modelGraph.nodesById[childId];
      if (isGroupNode(child) || (isOpNode(child) && !child.hideInLayout)) {
        descendants.push(child);
      }
      if (isGroupNode(child)) {
        this.gatherDescendants(modelGraph, child, descendants);
      }
    }
  }

  private processAttrValue(
    key: string,
    value: NodeAttributeValue,
  ): NodeAttributeValue {
    if (typeof value === 'string') {
      // Process const value that in `dense<...>` format. This is for backward
      // compatibility.
      if (value.startsWith('dense<')) {
        const matches = value.match(CONST_VALUE_REGEX);
        if (matches != null && matches.length > 1) {
          const strTensorValue = matches[1];
          return formatTensorValues(strTensorValue);
        }
      }
      // Process tensor values.
      else if (key === TENSOR_VALUES_KEY) {
        return formatTensorValues(value);
      }
      return value.replaceAll('"', '') || '<empty>';
    } else {
      return value;
    }
  }

  private processMetadataList(metadataItems: MetadataItem[]) {
    const metadata: Record<string, KeyValuePairs> = {};
    for (const metadataItem of metadataItems) {
      const attrs: KeyValuePairs = {};
      for (const attr of metadataItem.attrs) {
        let key = attr.key;
        let value = attr.value;
        // Special handlings.
        if (key === 'tensor_shape') {
          key = 'shape';
          value = value
            .replace('tensor<', '')
            .replace('>', '')
            .replace('*', '∗')
            .replace(/(\d|\?)x/g, '$1 x ');
        }
        attrs[key] = value;
      }
      metadata[metadataItem.id] = attrs;
    }
    return metadata;
  }

  private hasShapeTensorTag(
    metadata?: Record<string, KeyValuePairs>,
  ): boolean {
    return Object.values(metadata || {}).some((attrs) =>
      this.isShapeTensorTag(attrs[TENSOR_TAG_METADATA_KEY]),
    );
  }

  private shouldHideShapeNode(node: OpNode): boolean {
    const graphDomain = node.attrs?.['graph_domain'];
    if (graphDomain === 'shape') {
      return true;
    }
    if (graphDomain === 'data' || graphDomain === 'residual') {
      return false;
    }

    return this.hasShapeTensorTag(node.outputsMetadata);
  }

  private isShapeTensorTag(value: NodeAttributeValue | undefined): boolean {
    if (typeof value !== 'string') {
      return false;
    }
    const normalizedValue = value.toLowerCase();
    return normalizedValue === 'shape' || normalizedValue === 'tensor_shape';
  }
}

interface CompactArchitectureNode {
  node: GraphNode;
  firstSeenIndex: number;
}

interface ArchitectureFamily {
  familyKey: string;
  familyParts: string[];
  iterationNamespaces: string[];
  nodesByIterationNamespace: Map<string, GraphNode[]>;
}

interface ArchitectureIterationSignature {
  signature: string;
  nodeIdsInOrder: string[];
}

interface ArchitectureRun {
  runId: string;
  familyParts: string[];
  iterationNamespaces: string[];
  representativeIterationNamespace: string;
  label: string;
}

interface ArchitectureMembership {
  runId: string;
  iterationNamespace: string;
}

function buildArchitectureGraph(graph: Graph): Graph {
  const families = collectArchitectureFamilies(graph);
  if (families.length === 0) {
    return graph;
  }

  const directNodeIdMapping = new Map<string, string>();
  const runsByRepresentativeIterationKey = new Map<string, ArchitectureRun>();
  const runByIterationNamespace = new Map<string, ArchitectureRun>();

  for (const family of [...families].sort(
    (lhs, rhs) => rhs.familyParts.length - lhs.familyParts.length,
  )) {
    const signaturesByIterationNamespace = new Map<
      string,
      ArchitectureIterationSignature
    >();
    for (const iterationNamespace of family.iterationNamespaces) {
      signaturesByIterationNamespace.set(
        iterationNamespace,
        buildArchitectureIterationSignature(
          iterationNamespace,
          family.nodesByIterationNamespace.get(iterationNamespace) || [],
        ),
      );
    }

    for (let startIndex = 0; startIndex < family.iterationNamespaces.length; ) {
      const representativeIterationNamespace = family.iterationNamespaces[startIndex];
      const representativeSignature = signaturesByIterationNamespace.get(
        representativeIterationNamespace,
      );
      if (!representativeSignature) {
        startIndex++;
        continue;
      }

      let endIndex = startIndex + 1;
      while (endIndex < family.iterationNamespaces.length) {
        const nextSignature = signaturesByIterationNamespace.get(
          family.iterationNamespaces[endIndex],
        );
        if (nextSignature?.signature !== representativeSignature.signature) {
          break;
        }
        endIndex++;
      }

      const runCount = endIndex - startIndex;
      if (runCount > 1) {
        const runIterationNamespaces = family.iterationNamespaces.slice(
          startIndex,
          endIndex,
        );
        const run: ArchitectureRun = {
          runId: `${family.familyKey}|${runIterationNamespaces[0]}|${
            runIterationNamespaces[runIterationNamespaces.length - 1]
          }`,
          familyParts: family.familyParts,
          iterationNamespaces: runIterationNamespaces,
          representativeIterationNamespace,
          label: getArchitectureRunLabel(
            family.familyParts,
            representativeIterationNamespace,
            family.nodesByIterationNamespace.get(representativeIterationNamespace) ||
              [],
            runCount,
          ),
        };
        runsByRepresentativeIterationKey.set(
          getArchitectureRunKey(
            family.familyParts,
            getIterationTokenFromNamespace(representativeIterationNamespace),
          ),
          run,
        );
        for (const iterationNamespace of run.iterationNamespaces) {
          runByIterationNamespace.set(iterationNamespace, run);
        }

        for (let index = startIndex + 1; index < endIndex; index++) {
          const iterationNamespace = family.iterationNamespaces[index];
          const iterationSignature = signaturesByIterationNamespace.get(
            iterationNamespace,
          );
          if (!iterationSignature) {
            continue;
          }
          for (
            let nodeIndex = 0;
            nodeIndex < iterationSignature.nodeIdsInOrder.length;
            nodeIndex++
          ) {
            directNodeIdMapping.set(
              iterationSignature.nodeIdsInOrder[nodeIndex],
              representativeSignature.nodeIdsInOrder[nodeIndex],
            );
          }
        }
      }

      startIndex = endIndex;
    }
  }

  if (directNodeIdMapping.size === 0) {
    return graph;
  }

  const originalNodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const resolvedNodeIdCache = new Map<string, string>();
  const compactNodesById: Record<string, CompactArchitectureNode> = {};

  for (let index = 0; index < graph.nodes.length; index++) {
    const graphNode = graph.nodes[index];
    const compactNodeId = resolveArchitectureNodeId(
      graphNode.id,
      directNodeIdMapping,
      resolvedNodeIdCache,
    );
    if (compactNodeId !== graphNode.id) {
      continue;
    }
    compactNodesById[compactNodeId] = {
      node: cloneGraphNodeForArchitecture(
        graphNode,
        getCompactArchitectureNamespace(
          graphNode.namespace,
          runsByRepresentativeIterationKey,
        ),
      ),
      firstSeenIndex: getTopologicalIndex(graphNode, index),
    };
  }

  const seenIncomingEdges = new Set<string>();
  for (const graphNode of graph.nodes) {
    const originalTargetNode = originalNodesById.get(graphNode.id);
    if (!originalTargetNode) {
      continue;
    }
    const targetNodeId = resolveArchitectureNodeId(
      graphNode.id,
      directNodeIdMapping,
      resolvedNodeIdCache,
    );
    const compactTargetNode = compactNodesById[targetNodeId]?.node;
    if (!compactTargetNode) {
      continue;
    }
    for (const incomingEdge of graphNode.incomingEdges || []) {
      const originalSourceNode = originalNodesById.get(incomingEdge.sourceNodeId);
      if (!originalSourceNode) {
        continue;
      }
      if (
        shouldDropArchitectureEdge(
          originalSourceNode.namespace,
          originalTargetNode.namespace,
          runByIterationNamespace,
        )
      ) {
        continue;
      }
      const sourceNodeId = resolveArchitectureNodeId(
        incomingEdge.sourceNodeId,
        directNodeIdMapping,
        resolvedNodeIdCache,
      );
      if (sourceNodeId === targetNodeId) {
        continue;
      }
      const compactIncomingEdge = remapArchitectureIncomingEdge(
        incomingEdge,
        graphNode.id,
        sourceNodeId,
        targetNodeId,
        originalNodesById,
      );
      const edgeKey = getArchitectureEdgeKey(targetNodeId, compactIncomingEdge);
      if (seenIncomingEdges.has(edgeKey)) {
        continue;
      }
      seenIncomingEdges.add(edgeKey);
      if (compactTargetNode.incomingEdges == null) {
        compactTargetNode.incomingEdges = [];
      }
      compactTargetNode.incomingEdges.push(compactIncomingEdge);
    }
  }

  return {
    ...graph,
    nodes: Object.values(compactNodesById)
      .sort((lhs, rhs) => lhs.firstSeenIndex - rhs.firstSeenIndex)
      .map((compactNode) => compactNode.node),
  };
}

function collectArchitectureFamilies(graph: Graph): ArchitectureFamily[] {
  const familiesByKey = new Map<string, ArchitectureFamily>();
  for (const graphNode of graph.nodes) {
    const namespaceParts = splitNamespace(graphNode.namespace);
    for (let index = 0; index < namespaceParts.length; index++) {
      const namespacePart = namespaceParts[index];
      if (!isNumericNamespacePart(namespacePart)) {
        continue;
      }
      const familyParts = namespaceParts.slice(0, index);
      const familyKey = [...familyParts, '#'].join('/');
      const iterationNamespace = namespaceParts.slice(0, index + 1).join('/');

      let family = familiesByKey.get(familyKey);
      if (!family) {
        family = {
          familyKey,
          familyParts,
          iterationNamespaces: [],
          nodesByIterationNamespace: new Map<string, GraphNode[]>(),
        };
        familiesByKey.set(familyKey, family);
      }

      if (!family.nodesByIterationNamespace.has(iterationNamespace)) {
        family.iterationNamespaces.push(iterationNamespace);
        family.nodesByIterationNamespace.set(iterationNamespace, []);
      }
      family.nodesByIterationNamespace.get(iterationNamespace)?.push(graphNode);
    }
  }

  return Array.from(familiesByKey.values())
    .filter((family) => family.iterationNamespaces.length > 1)
    .map((family) => ({
      ...family,
      iterationNamespaces: [...family.iterationNamespaces].sort(
        compareIterationNamespaces,
      ),
    }));
}

function buildArchitectureIterationSignature(
  iterationNamespace: string,
  iterationNodes: GraphNode[],
): ArchitectureIterationSignature {
  const orderedNodes = [...iterationNodes].sort((lhs, rhs) => {
    return getTopologicalIndex(lhs, 0) - getTopologicalIndex(rhs, 0);
  });
  const orderedNodeIds = orderedNodes.map((node) => node.id);
  const nodeOrderById = new Map(
    orderedNodeIds.map((nodeId, index) => [nodeId, index]),
  );
  const nodeDescriptors = orderedNodes.map((node) =>
    JSON.stringify({
      namespace: getRelativeArchitectureNamespace(node.namespace, iterationNamespace),
      label: node.label,
      target: normalizeArchitectureValue(getGraphNodeAttr(node, 'target')),
      modulePath: normalizeArchitectureValue(
        getGraphNodeAttr(node, 'module_path'),
      ),
    }),
  );
  const edgeDescriptors: string[] = [];
  for (const targetNode of orderedNodes) {
    const targetIndex = nodeOrderById.get(targetNode.id);
    if (targetIndex == null) {
      continue;
    }
    for (const incomingEdge of targetNode.incomingEdges || []) {
      const sourceIndex = nodeOrderById.get(incomingEdge.sourceNodeId);
      if (sourceIndex == null) {
        continue;
      }
      edgeDescriptors.push(
        [
          sourceIndex,
          targetIndex,
          incomingEdge.metadata?.['edge_color_role'] || '',
          incomingEdge.metadata?.['edge_is_residual'] || '',
          incomingEdge.metadata?.['edge_is_shape'] || '',
        ].join('|'),
      );
    }
  }

  edgeDescriptors.sort();
  return {
    signature: JSON.stringify({
      nodes: nodeDescriptors,
      edges: edgeDescriptors,
    }),
    nodeIdsInOrder: orderedNodeIds,
  };
}

function getArchitectureRunLabel(
  familyParts: string[],
  iterationNamespace: string,
  iterationNodes: GraphNode[],
  runCount: number,
): string {
  const familyLabel = unEscapeString(
    familyParts[familyParts.length - 1] || '',
  ).trim();
  if (familyLabel) {
    return `${familyLabel} x ${runCount}`;
  }

  const representativeNode = [...iterationNodes].sort((lhs, rhs) => {
    const depthDelta =
      getRelativeArchitectureDepth(lhs.namespace, iterationNamespace) -
      getRelativeArchitectureDepth(rhs.namespace, iterationNamespace);
    if (depthDelta !== 0) {
      return depthDelta;
    }
    return getTopologicalIndex(lhs, 0) - getTopologicalIndex(rhs, 0);
  })[0];
  const representativeLabel = representativeNode?.label?.trim() || 'Block';
  return `${representativeLabel} x ${runCount}`;
}

function resolveArchitectureNodeId(
  nodeId: string,
  directNodeIdMapping: Map<string, string>,
  resolvedNodeIdCache: Map<string, string>,
): string {
  const cachedNodeId = resolvedNodeIdCache.get(nodeId);
  if (cachedNodeId) {
    return cachedNodeId;
  }

  let resolvedNodeId = nodeId;
  const seenNodeIds = new Set<string>();
  while (directNodeIdMapping.has(resolvedNodeId)) {
    if (seenNodeIds.has(resolvedNodeId)) {
      break;
    }
    seenNodeIds.add(resolvedNodeId);
    resolvedNodeId = directNodeIdMapping.get(resolvedNodeId) || resolvedNodeId;
  }
  resolvedNodeIdCache.set(nodeId, resolvedNodeId);
  return resolvedNodeId;
}

function shouldDropArchitectureEdge(
  sourceNamespace: string,
  targetNamespace: string,
  runByIterationNamespace: Map<string, ArchitectureRun>,
): boolean {
  const sourceMembership = findArchitectureMembership(
    sourceNamespace,
    runByIterationNamespace,
  );
  const targetMembership = findArchitectureMembership(
    targetNamespace,
    runByIterationNamespace,
  );
  return (
    sourceMembership != null &&
    targetMembership != null &&
    sourceMembership.runId === targetMembership.runId &&
    sourceMembership.iterationNamespace !== targetMembership.iterationNamespace
  );
}

function findArchitectureMembership(
  namespace: string,
  runByIterationNamespace: Map<string, ArchitectureRun>,
): ArchitectureMembership | undefined {
  if (namespace === '') {
    return undefined;
  }
  const namespaceParts = splitNamespace(namespace);
  for (let depth = namespaceParts.length; depth > 0; depth--) {
    const iterationNamespace = namespaceParts.slice(0, depth).join('/');
    const run = runByIterationNamespace.get(iterationNamespace);
    if (!run) {
      continue;
    }
    return {
      runId: run.runId,
      iterationNamespace,
    };
  }
  return undefined;
}

function remapArchitectureIncomingEdge(
  incomingEdge: {
    sourceNodeId: string;
    sourceNodeOutputId: string;
    targetNodeInputId: string;
    metadata?: KeyValuePairs;
  },
  originalTargetNodeId: string,
  sourceNodeId: string,
  targetNodeId: string,
  originalNodesById: Map<string, GraphNode>,
): {
  sourceNodeId: string;
  sourceNodeOutputId: string;
  targetNodeInputId: string;
  metadata?: KeyValuePairs;
} {
  const originalSourceNode = originalNodesById.get(incomingEdge.sourceNodeId);
  const originalTargetNode = originalNodesById.get(originalTargetNodeId);
  const mappedSourceNode = originalNodesById.get(sourceNodeId);
  const mappedTargetNode = originalNodesById.get(targetNodeId);

  return {
    ...incomingEdge,
    sourceNodeId,
    sourceNodeOutputId: remapArchitectureMetadataId(
      originalSourceNode,
      mappedSourceNode,
      incomingEdge.sourceNodeOutputId,
      'output',
    ),
    targetNodeInputId: remapArchitectureMetadataId(
      originalTargetNode,
      mappedTargetNode,
      incomingEdge.targetNodeInputId,
      'input',
    ),
  };
}

function remapArchitectureMetadataId(
  originalNode: GraphNode | undefined,
  mappedNode: GraphNode | undefined,
  metadataId: string,
  kind: 'input' | 'output',
): string {
  if (!originalNode || !mappedNode || originalNode.id === mappedNode.id) {
    return metadataId;
  }

  const originalMetadataIds = getArchitectureMetadataIds(originalNode, kind);
  const mappedMetadataIds = getArchitectureMetadataIds(mappedNode, kind);
  const metadataIndex = originalMetadataIds.indexOf(metadataId);
  if (metadataIndex < 0 || metadataIndex >= mappedMetadataIds.length) {
    return metadataId;
  }
  return mappedMetadataIds[metadataIndex];
}

function getArchitectureMetadataIds(
  graphNode: GraphNode,
  kind: 'input' | 'output',
): string[] {
  return (kind === 'input' ? graphNode.inputsMetadata : graphNode.outputsMetadata)
    ?.map((metadataItem) => metadataItem.id) || [];
}

function getCompactArchitectureNamespace(
  namespace: string,
  runsByRepresentativeIterationKey: Map<string, ArchitectureRun>,
): string {
  if (namespace === '') {
    return '';
  }

  const namespaceParts = splitNamespace(namespace);
  const compactNamespaceParts: string[] = [];
  for (let index = 0; index < namespaceParts.length; index++) {
    const namespacePart = namespaceParts[index];
    if (isNumericNamespacePart(namespacePart)) {
      const familyParts = namespaceParts.slice(0, index);
      const run = runsByRepresentativeIterationKey.get(
        getArchitectureRunKey(familyParts, namespacePart),
      );
      if (run) {
        const escapedLabel = escapeArchitectureNamespaceSegment(run.label);
        if (compactNamespaceParts.length > 0) {
          compactNamespaceParts[compactNamespaceParts.length - 1] = escapedLabel;
        } else {
          compactNamespaceParts.push(escapedLabel);
        }
        continue;
      }
    }
    compactNamespaceParts.push(namespacePart);
  }

  return compactNamespaceParts.join('/');
}

function cloneGraphNodeForArchitecture(
  graphNode: GraphNode,
  compactNamespace: string,
): GraphNode {
  return {
    ...graphNode,
    namespace: compactNamespace,
    incomingEdges: [],
    attrs: cloneNodeAttributeList(graphNode.attrs),
    inputsMetadata: cloneMetadataList(graphNode.inputsMetadata),
    outputsMetadata: cloneMetadataList(graphNode.outputsMetadata),
  };
}

function cloneNodeAttributeList(
  attrs: NodeAttributeList | undefined,
): NodeAttributeList | undefined {
  return attrs?.map((attr) => ({...attr}));
}

function cloneMetadataList(
  metadataList: MetadataItem[] | undefined,
): MetadataItem[] | undefined {
  return metadataList?.map((metadataItem) => ({
    ...metadataItem,
    attrs: metadataItem.attrs.map((attr) => ({...attr})),
  }));
}

function getArchitectureEdgeKey(
  targetNodeId: string,
  incomingEdge: {
    sourceNodeId: string;
    sourceNodeOutputId: string;
    targetNodeInputId: string;
    metadata?: KeyValuePairs;
  },
): string {
  return [
    incomingEdge.sourceNodeId,
    targetNodeId,
    incomingEdge.metadata?.['value'] || '',
    incomingEdge.metadata?.['edge_color_role'] || '',
    incomingEdge.metadata?.['edge_is_residual'] || '',
    incomingEdge.metadata?.['edge_is_shape'] || '',
  ].join('|');
}

function compareIterationNamespaces(lhs: string, rhs: string): number {
  return getIterationIndexFromNamespace(lhs) - getIterationIndexFromNamespace(rhs);
}

function getIterationIndexFromNamespace(iterationNamespace: string): number {
  return Number.parseInt(getIterationTokenFromNamespace(iterationNamespace), 10);
}

function getArchitectureRunKey(
  familyParts: string[],
  iterationToken: string,
): string {
  return `${familyParts.join('/')}|${iterationToken}`;
}

function getIterationTokenFromNamespace(iterationNamespace: string): string {
  const namespaceParts = splitNamespace(iterationNamespace);
  return namespaceParts[namespaceParts.length - 1] || '0';
}

function getRelativeArchitectureNamespace(
  namespace: string,
  iterationNamespace: string,
): string {
  const namespaceParts = splitNamespace(namespace);
  const iterationNamespaceParts = splitNamespace(iterationNamespace);
  return namespaceParts.slice(iterationNamespaceParts.length).join('/');
}

function getRelativeArchitectureDepth(
  namespace: string,
  iterationNamespace: string,
): number {
  return splitNamespace(
    getRelativeArchitectureNamespace(namespace, iterationNamespace),
  ).length;
}

function getGraphNodeAttr(
  graphNode: GraphNode,
  attrKey: string,
): string {
  const attrValue = graphNode.attrs?.find((attr) => attr.key === attrKey)?.value;
  return typeof attrValue === 'string' ? attrValue : '';
}

function normalizeArchitectureValue(value: string): string {
  return value.replace(/(^|[./])(\d+)(?=([./]|$))/g, '$1#');
}

function isNumericNamespacePart(namespacePart: string): boolean {
  return /^\d+$/.test(namespacePart);
}

function escapeArchitectureNamespaceSegment(segment: string): string {
  return segment.replace(/\//g, '\\/').trim() || 'Block';
}

function getTopologicalIndex(graphNode: GraphNode, fallbackIndex: number): number {
  const topologicalIndex = graphNode.attrs?.find(
    (attr) => attr.key === 'topological_index',
  )?.value;
  if (typeof topologicalIndex === 'string') {
    const parsed = Number.parseInt(topologicalIndex, 10);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return fallbackIndex;
}

function getOutgoingEdgeStyleData(edge: OutgoingEdge): EdgeStyleData {
  const colorRole = parseEdgeColorRole(edge.metadata?.['edge_color_role']);
  const legacyEdgeDomain = edge.metadata?.['edge_domain'];
  return {
    colorRole,
    isResidual:
      parseEdgeBoolean(edge.metadata?.['edge_is_residual']) ||
      legacyEdgeDomain === 'residual',
    isShape:
      parseEdgeBoolean(edge.metadata?.['edge_is_shape']) ||
      legacyEdgeDomain === 'shape',
  };
}

function mergeEdgeStyleData(
  currentEdgeStyle: EdgeStyleData | undefined,
  nextEdgeStyle: EdgeStyleData,
): EdgeStyleData {
  if (!currentEdgeStyle) {
    return nextEdgeStyle;
  }
  return {
    colorRole: pickPreferredColorRole(
      currentEdgeStyle.colorRole,
      nextEdgeStyle.colorRole,
    ),
    isResidual: currentEdgeStyle.isResidual || nextEdgeStyle.isResidual,
    isShape: currentEdgeStyle.isShape || nextEdgeStyle.isShape,
  };
}

function parseEdgeColorRole(value: NodeAttributeValue | undefined): EdgeColorRole {
  switch (value) {
    case 'input':
    case 'output':
      return value;
    default:
      return 'data';
  }
}

function parseEdgeBoolean(value: NodeAttributeValue | undefined): boolean {
  return value === true || value === 'true';
}

function pickPreferredColorRole(
  currentColorRole: EdgeColorRole,
  nextColorRole: EdgeColorRole,
): EdgeColorRole {
  const priority: Record<EdgeColorRole, number> = {
    input: 3,
    output: 2,
    data: 1,
  };
  return priority[nextColorRole] > priority[currentColorRole]
    ? nextColorRole
    : currentColorRole;
}

/**
 * Formats the given tensor values string.
 *
 * The given string is in the form of:
 * [[[1, 2], [3, 4]]]
 *
 * And we want to format it to:
 * [
 *   [
 *     [
 *       1,
 *       2
 *     ],
 *     [
 *       3,
 *       4
 *     ]
 *   ]
 * ]
 */
export function formatTensorValues(strValues: string): string {
  try {
    return JSON.stringify(JSON.parse(strValues), null, 2)
      .replaceAll('\\n', '\n')
      .trim();
  } catch (e) {
    return strValues;
  }
}
