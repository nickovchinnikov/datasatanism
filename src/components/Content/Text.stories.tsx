import React from "react";
import { StoryObj, Meta } from "@storybook/react";

import {
  Heading1,
  Heading2,
  Heading3,
  Heading4,
  Heading5,
  Heading6,
  Paragraph,
  Span,
} from "./Text";

const meta = {
  title: "Components/Text",
  component: Paragraph,
  args: {
    children: "This is a paragraph",
  },
} satisfies Meta<typeof Paragraph>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ParagraphStory: Story = {
  args: {
    children: "This is a paragraph",
  },
  render: (args) => <Paragraph>{args.children}</Paragraph>,
};

export const Heading1Story: Story = {
  args: {
    children: "Heading 1",
  },
  render: (args) => <Heading1>{args.children}</Heading1>,
};

export const Heading2Story: Story = {
  args: {
    children: "Heading 2",
  },
  render: (args) => <Heading2>{args.children}</Heading2>,
};

export const Heading3Story: Story = {
  args: {
    children: "Heading 3",
  },
  render: (args) => <Heading3>{args.children}</Heading3>,
};

export const Heading4Story: Story = {
  args: {
    children: "Heading 4",
  },
  render: (args) => <Heading4>{args.children}</Heading4>,
};

export const Heading5Story: Story = {
  args: {
    children: "Heading 5",
  },
  render: (args) => <Heading5>{args.children}</Heading5>,
};

export const Heading6Story: Story = {
  args: {
    children: "Heading 6",
  },
  render: (args) => <Heading6>{args.children}</Heading6>,
};

export const SpanStory: Story = {
  args: {
    children: "Span Text",
  },
  render: (args) => <Span>{args.children}</Span>,
};
