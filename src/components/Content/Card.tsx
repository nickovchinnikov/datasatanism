import { FC, ReactNode } from "react";

import { Pad } from "./Pad";
import { Heading4, Paragraph } from "./Text";

interface Blocks {
  key: string;
  value: ReactNode;
}

interface Props {
  head: string;
  blocks?: Blocks[];
}

export const Card: FC<Props> = ({ head, blocks }) => (
  <Pad>
    <Heading4>{head}</Heading4>
    {blocks?.map((block) => (
      <Paragraph key={block.key}>{block.value}</Paragraph>
    ))}
  </Pad>
);
