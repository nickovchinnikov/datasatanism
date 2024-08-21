import type { Meta, StoryObj } from "@storybook/react";

import { Loading } from "./Loading";

const meta = {
  title: "Pages/Loading",
  component: Loading,
} satisfies Meta<typeof Loading>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    systemMsg: "Download and install the model",
    systemInf: "LLAMA 3.1 8B params 7b quantized",
    successMsg: "Installed: LLAMA 3.1 8B"
  },
};
