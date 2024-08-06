import { FC } from "react";

import Button from "@/components/Button";
import { Icon, AvailableIcons } from "@/components/Icons/Icon";

interface Props {
  icon: AvailableIcons;
  size?: number;
  onClick: () => void;
}

export const IconButton: FC<Props> = ({ icon, size = 2, onClick }) => {
  return (
    <Button variant="icon" onClick={onClick}>
      <Icon name={icon} size={size} />
    </Button>
  );
};
