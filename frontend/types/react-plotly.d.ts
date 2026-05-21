declare module "react-plotly.js" {
  import type { ComponentType } from "react";
  import type { Layout, PlotData, Config } from "plotly.js";

  export type PlotParams = {
    data: Partial<PlotData>[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    className?: string;
    useResizeHandler?: boolean;
    style?: React.CSSProperties;
  };

  const Plot: ComponentType<PlotParams>;
  export default Plot;
}
