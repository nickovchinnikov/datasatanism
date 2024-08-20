import type { Meta, StoryObj } from "@storybook/react";

import { Caution } from "./Caution";

const meta: Meta<typeof Caution> = {
  title: "Msg/Caution",
  component: Caution,
  args: {},
} satisfies Meta<typeof Caution>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "CAUTION",
  },
};
