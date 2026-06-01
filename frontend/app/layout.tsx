import type { Metadata } from "next";
import type { ReactNode } from "react";
import { AuthProvider } from "@/components/AuthProvider";
import { PostHogProvider } from "@/components/PostHogProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroPrice",
  description: "PINN-powered option pricing SaaS MVP",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" translate="no" suppressHydrationWarning>
      <body><PostHogProvider><AuthProvider>{children}</AuthProvider></PostHogProvider></body>
    </html>
  );
}
