import { FC, useEffect, useRef } from "react";

import styles from "./Background.module.scss";

export const Background: FC = () => {
  const lightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = ({ clientX: x, clientY: y }: MouseEvent) => {
      if (lightRef.current) {
        lightRef.current.style.background = `radial-gradient(circle at ${x}px ${y}px, transparent 0%, #000 30%)`;
      }
    };

    document.addEventListener("mousemove", handleMouseMove);

    // Clean up event listener on component unmount
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.light} ref={lightRef} />
    </div>
  );
};
