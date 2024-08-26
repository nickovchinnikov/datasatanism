
import type { Meta, StoryObj } from "@storybook/react";

import { Nav } from "./Nav";
import { Props as ItemProps } from "./Item";

const meta: Meta<typeof Nav> = {
  title: "Navigation/Nav",
  component: Nav,
  args: {},
} satisfies Meta<typeof Nav>;

export default meta;
type Story = StoryObj<typeof meta>;

const items: ItemProps[] = [
    {
        name: "qwen2-1_5b-instruct-q4_k_m",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    },
    {
        name: "gemma-2-2b-it-abliterated-Q4_K_M",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    },
    {
        name: "neuralreyna-mini-1.8b-v0.3.q4_k_m",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    },
    {
        name: "Phi-3.1-mini-128k-instruct-Q3_K_M",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    },
    {
        name: "meta-llama-3.1-8b-instruct-abliterated.Q2_K",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    },
    {
        name: "Meta-Llama-3.1-8B-Instruct-Q2_K",
        unread: 0,
        active: false,
        type: "direct",
        red: true
    }
]

export const Primary: Story = {
  args: {
    items
  },
};
