# V21 Case Tree Figure Index

English case-level causal and component trees generated from the case-local CSV files. The high-resolution rebuild uses a card-based layered layout inspired by the earlier causal-KG extraction and ATA component-tree extraction workflows, but it reads only the V21 package artifacts. Each figure has PNG/SVG/PDF, source plot data, and caption.

## Boeing causal KG tree

Shows condition, threshold, warning/action, and component-anchor flows aggregated from Boeing causal KG edges.

![Boeing causal KG tree](../../Cases/Boeing_737/2%20因果关系KG抽取-step2.1/result_trees_highres/Boeing_737_diagnostic_causal_tree_highres.png)

## Boeing component-role tree

Shows how Boeing evidence objects are organized by ATA-oriented component family, profile anchors, and diagnostic role.

![Boeing component-role tree](../../Cases/Boeing_737/3%20组件结构的抽取-step2.2/result_trees_highres/Boeing_737_component_role_tree_highres.png)

## Cessna causal KG tree

Shows the richer Cessna procedure/causal graph across condition, warning, cause, inspection, replacement, and component anchors.

![Cessna causal KG tree](../../Cases/Cessna_172/2%20因果关系KG抽取-step2.1/result_trees_highres/Cessna_172_diagnostic_causal_tree_highres.png)

## Cessna component-role tree

Shows the Cessna component evidence index and why package retrieval benefits from component-role constraints.

![Cessna component-role tree](../../Cases/Cessna_172/3%20组件结构的抽取-step2.2/result_trees_highres/Cessna_172_component_role_tree_highres.png)

## Rebuild code

- Root script: `10_scripts/build_highres_case_trees.py`
- Boeing code copy: `Cases/Boeing_737/tree_building_code/build_highres_case_trees.py`
- Cessna code copy: `Cases/Cessna_172/tree_building_code/build_highres_case_trees.py`
