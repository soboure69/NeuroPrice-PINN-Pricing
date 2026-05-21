"use client";

import { useEffect, useRef } from "react";
import type { Config, Layout, PlotData } from "plotly.js";

type PriceSurfacePlotProps = {
  spotGrid: number[];
  maturityGrid: number[];
  z: number[][];
};

export function PriceSurfacePlot({ spotGrid, maturityGrid, z }: PriceSurfacePlotProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const hasPlot = useRef(false);

  useEffect(() => {
    let mounted = true;

    async function renderPlot() {
      const Plotly = await import("plotly.js-dist-min");
      if (!mounted || !ref.current) {
        return;
      }

      const data: Partial<PlotData>[] = [
        {
          x: spotGrid,
          y: maturityGrid,
          z,
          type: "surface",
          colorscale: "Viridis",
        },
      ];

      const layout: Partial<Layout> = {
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e2e8f0" },
        margin: { l: 0, r: 0, t: 10, b: 0 },
        scene: {
          xaxis: { title: { text: "S" } },
          yaxis: { title: { text: "T" } },
          zaxis: { title: { text: "Price" } },
        },
      };

      const config: Partial<Config> = {
        displayModeBar: false,
        responsive: true,
      };

      if (hasPlot.current) {
        Plotly.react(ref.current, data, layout, config);
      } else {
        Plotly.newPlot(ref.current, data, layout, config);
        hasPlot.current = true;
      }
    }

    renderPlot();

    return () => {
      mounted = false;
      const node = ref.current;
      if (node) {
        import("plotly.js-dist-min").then((Plotly) => {
          try {
            Plotly.purge(node);
            hasPlot.current = false;
          } catch {
            hasPlot.current = false;
          }
        });
      }
    };
  }, [maturityGrid, spotGrid, z]);

  return <div ref={ref} className="h-[420px] w-full" />;
}
