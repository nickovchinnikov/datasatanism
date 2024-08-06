export const randNumber = (min: number, max: number): number =>
  Math.floor(Math.random() * (max - min + 1) + min);

export const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

export const typeString = async function* (
  text: string,
): AsyncGenerator<string> {
  for (const char of text.split("")) {
    const ms = randNumber(50, 150);
    await delay(ms);
    yield char;
  }
};

export const timesAgo = (datetime: string) => {
  const currentDate = new Date();
  const messageDate = new Date(datetime);

  const diff = currentDate.getTime() - messageDate.getTime();

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return <span>{days} days ago</span>;
  } else if (hours > 0) {
    return <span>{hours} hours ago</span>;
  } else if (minutes > 0) {
    return <span>{minutes} minutes ago</span>;
  } else {
    return <span>now</span>;
  }
};
