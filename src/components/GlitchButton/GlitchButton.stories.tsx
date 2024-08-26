import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { GlitchButton } from "./GlitchButton";

const meta: Meta<typeof GlitchButton> = {
  title: "Buttons/GlitchButton",
  component: GlitchButton,
  args: { onClick: fn() },
} satisfies Meta<typeof GlitchButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "Download"
  },
};
