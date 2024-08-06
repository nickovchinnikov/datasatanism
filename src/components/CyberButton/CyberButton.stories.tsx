import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { CyberButton } from "./CyberButton";

const meta = {
  title: "Components/CyberButton",
  component: CyberButton,
  args: { onClick: fn() },
} satisfies Meta<typeof CyberButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "Button",
    type: "button",
    onClick: fn(),
  },
};
