import { FC } from "react";
// import Image from "next/image";

import styles from "./CircleImage.module.scss";

interface Props {
  src: string;
  isActive?: boolean;
  circles?: number;
  alt?: string;
  size?: string;
}

export const CircleImage: FC<Props> = ({
  src,
  isActive = false,
  circles = 4,
  alt = "",
  size = "150px",
}) => (
  <div className={styles.image} style={{ width: size, height: size }}>
    <div className={styles.contain}>
      <div
        className={styles.circles}
        style={{ display: isActive ? "block" : "none" }}
      >
        {Array(circles)
          .fill(null)
          .map((_, index) => (
            <div key={index} className={styles.circle} />
          ))}
      </div>
      <img
        src={src}
        alt={alt}
        width={0}
        height={0}
        sizes="100vw"
        style={{ width: "100%", height: "auto" }}
      />
    </div>
  </div>
);
