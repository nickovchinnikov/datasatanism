import type { Meta, StoryObj } from "@storybook/react";

import { CircleImage } from "./CircleImage";

const meta = {
  title: "Components/CircleImage",
  component: CircleImage,
} satisfies Meta<typeof CircleImage>;

export default meta;
export const Default: StoryObj<typeof CircleImage> = {
  args: {
    src: "./mock_image.png",
    size: "150px",
    alt: "mock image",
    isActive: true,
  },
};
