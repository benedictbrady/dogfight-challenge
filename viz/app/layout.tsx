import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dogfight Visualizer",
  description: "3D visualization for dogfight challenge matches",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#0a0a0a] text-gray-200 antialiased">
        {children}
      </body>
    </html>
  );
}
