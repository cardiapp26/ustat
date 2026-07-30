const columnStructureInFlight = new Set<string>();

export async function runColumnStructureMutation<T>(
  sessionId: string,
  mutation: () => Promise<T>,
): Promise<T> {
  if (columnStructureInFlight.has(sessionId)) {
    throw new Error("Another column change is already in progress.");
  }
  columnStructureInFlight.add(sessionId);
  try {
    return await mutation();
  } finally {
    columnStructureInFlight.delete(sessionId);
  }
}
