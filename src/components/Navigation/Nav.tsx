import { FC } from "react";

import { CyberButton } from "@/components/CyberButton/CyberButton"

import { Item, Props as ItemProps } from "./Item"

interface Props {
    items: ItemProps[]
}

export const Nav: FC<Props> = ({ items }) => <>
    <h2>Navigation</h2>
    {items.map((item: ItemProps) => <><Item key={item.name} {...item} /></>)}
</>