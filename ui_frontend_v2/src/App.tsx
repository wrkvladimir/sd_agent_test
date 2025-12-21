import { KnowledgeCard } from "@/features/knowledge/knowledge-card";
import { ScenarioCard } from "@/features/scenarios/scenario-card";
import { SystemConfigCard } from "@/features/system/system-config-card";
import { ChatInstances } from "@/features/chat/chat-instances";

function App() {
  return (
    <div className="h-screen overflow-hidden bg-background">
      <div className="flex h-screen gap-3 p-3">
        <aside className="flex w-[440px] min-w-[440px] flex-col gap-3 overflow-hidden">
          <div className="grid min-h-0 flex-1 grid-rows-3 gap-3">
            <KnowledgeCard className="min-h-0" />
            <ScenarioCard className="min-h-0" />
            <SystemConfigCard className="min-h-0" />
          </div>
        </aside>

        <main className="min-w-0 flex-1 overflow-hidden">
          <ChatInstances className="h-full" />
        </main>
      </div>
    </div>
  );
}

export default App



