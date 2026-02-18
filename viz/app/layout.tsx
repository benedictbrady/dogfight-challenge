import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dogfight Visualizer",
  description: "2D visualization for dogfight challenge matches",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#f8f8f8] text-gray-800 antialiased">
        {children}
      </body>
    </html>
  );
}
