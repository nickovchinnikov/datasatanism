import type { Meta, StoryObj } from "@storybook/react";

import { Init } from "./Init";

const meta: Meta<typeof Init> = {
  title: "Modules/Init",
  component: Init,
  args: {},
} satisfies Meta<typeof Init>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    glitch: true,
    success: false,
  },
};
