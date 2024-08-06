import { FC, ReactNode, MouseEvent } from "react";

import style from "./CyberButton.module.scss";

interface Props {
  children: ReactNode;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  type?: "button" | "submit" | "reset";
}

export const CyberButton: FC<Props> = ({
  children,
  onClick,
  type = "button",
}) => {
  return (
    <button className={style.button} type={type} onClick={onClick}>
      {children}
    </button>
  );
};
