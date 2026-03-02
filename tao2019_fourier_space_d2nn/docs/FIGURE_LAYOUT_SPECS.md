# Figure Layout Specs (pixel-precise)

Layout engine implementation: `src/tao2019_fd2nn/viz/layout_specs.py`.

## Coordinate convention
- bbox format: `(x_px, y_px, w_px, h_px)`
- origin: top-left
- renderer converts to `matplotlib.add_axes(left,bottom,width,height)`

## `layout_id="fig2"`
- canvas: `1800x1400 @ 300dpi`
- panel grid: `4x5`, panel size `300x300`
- row positions: `80, 390, 700, 1010`
- column positions: `150, 460, 770, 1080, 1390`
- colorbars: `x=1700`, width `40`, per-row + phase row
- text anchors include row labels and cell type titles

## `layout_id="fig3"`
- same geometry as `fig2`
- row0 label changed to `"Dynamic Scene"`
- cell type titles removed

## `layout_id="fig4"`
- canvas: `2600x1600 @ 300dpi`
- subfigure (a) and (b) split
- top config/image panels + bottom convergence axes
- fixed bboxes for:
  - `a_cfg0..a_cfg3`, `a_img_r{r}_c{c}`, `a_plot`
  - `b_cfg0..b_cfg3`, `b_plot`
- figure-level text anchors include `(a)` and `(b)` labels
