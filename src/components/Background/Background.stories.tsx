import type { Meta, StoryObj } from "@storybook/react";

import { Background } from "./Background";

const meta = {
  title: "Components/Background",
  component: Background,
} satisfies Meta<typeof Background>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {},
};
