# V21 High-Resolution Case Tree Rebuild Report

Generated on 2026-05-15.

## Motivation

The previous case-tree figures used compact node-link layouts. They were serviceable for internal inspection, but the labels were too dense for results discussion and some text collided with visual elements after export. The new rebuild uses a wide layered card layout with larger figure canvases, wrapped English labels, and case-local source data.

## Reference Workflows

The redesign follows the methodology of the earlier backup workflows:

- Causal KG extraction: `4月23生成的新代码(2)_BACKUP_20260507_234823/2 因果关系KG抽取-step2.1`
- ATA component structure extraction: `4月23生成的新代码(2)_BACKUP_20260507_234823/3 组件结构的抽取-step2.2/ata_component`

The new script does not rerun those historical pipelines. It uses the V21 case-local causal-edge, evidence-object, component-profile, and alias files so the figures remain tied to the runnable V21 package.

## Generated Figures

For each case, two English high-resolution trees were rebuilt:

- Diagnostic causal KG tree: case root -> causal relation -> mechanism card -> component anchors.
- ATA-oriented component-role tree: case root -> component family -> profile anchors -> diagnostic roles.

Each figure is exported as:

- `PNG` at 600 dpi;
- `SVG` with editable text;
- `PDF` with embedded TrueType fonts;
- `*_plot_data.csv`;
- `*_caption.md`.

## Case-Local Locations

### Boeing 737

- `Cases/Boeing_737/2 因果关系KG抽取-step2.1/result_trees_highres/`
- `Cases/Boeing_737/3 组件结构的抽取-step2.2/result_trees_highres/`
- Mirrored for result browsing: `Cases/Boeing_737/7 测试/result_trees_highres/`
- Code copy: `Cases/Boeing_737/tree_building_code/build_highres_case_trees.py`

### Cessna 172

- `Cases/Cessna_172/2 因果关系KG抽取-step2.1/result_trees_highres/`
- `Cases/Cessna_172/3 组件结构的抽取-step2.2/result_trees_highres/`
- Mirrored for result browsing: `Cases/Cessna_172/7 测试/result_trees_highres/`
- Code copy: `Cases/Cessna_172/tree_building_code/build_highres_case_trees.py`

## Visual QA

The PNG exports were verified after generation:

- Boeing causal tree: 8862 x 3585 px.
- Boeing component-role tree: 9234 x 5968 px.
- Cessna causal tree: 8862 x 6070 px.
- Cessna component-role tree: 9234 x 6532 px.

The new design separates text cards horizontally and vertically to avoid overlap. The SVG/PDF versions should be preferred for paper editing.

## Claim Boundary

These trees are result-explanation figures for document-grounded causal and component evidence structures. They are not field-validated aircraft causal models and do not imply certified maintenance diagnosis.

