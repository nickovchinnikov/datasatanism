import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Loader } from "./Loader";

const meta: Meta<typeof Loader> = {
  title: "Components/Loader",
  component: Loader,
  args: {},
} satisfies Meta<typeof Loader>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    width: "30vw",
    height: "30vh",
  },
};
