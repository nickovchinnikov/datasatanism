import type { Meta, StoryObj } from "@storybook/react";
import { useWllama, WllamaProvider } from '@/utils/wllama.context';

import { Models } from "./Models";

const meta: Meta<typeof Models> = {
  title: "Pages/Models",
  component: Models,
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {},
  decorators: [
    (Story) => (
      <WllamaProvider>
        <Story />
      </WllamaProvider>
    ),
  ],
};
