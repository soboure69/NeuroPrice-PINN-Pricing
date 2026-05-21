declare module "plotly.js-dist-min" {
  import type { Config, Layout, PlotData, Root } from "plotly.js";

  export function newPlot(
    root: Root,
    data: Partial<PlotData>[],
    layout?: Partial<Layout>,
    config?: Partial<Config>,
  ): Promise<Root>;

  export function react(
    root: Root,
    data: Partial<PlotData>[],
    layout?: Partial<Layout>,
    config?: Partial<Config>,
  ): Promise<Root>;

  export function purge(root: Root): void;
}
