import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { FullSlug, resolveRelative } from "../util/path"

const Header: QuartzComponent = ({ children, fileData }: QuartzComponentProps) => {
  const currentSlug = fileData.slug?.replace(/\/index$/, "").toLowerCase() ?? ""
  const links: [string, FullSlug][] = [
    ["Atlas", "index" as FullSlug],
    ["学习地图", "02-Learning-Map/index" as FullSlug],
    ["CS224N", "03-CS224N/index" as FullSlug],
    ["CS336", "04-CS336/index" as FullSlug],
    ["开源讲义", "06-Hands-on-LLM/index" as FullSlug],
    ["参考", "05-Reference/Glossary" as FullSlug],
  ]
  return (
    <header class="atlas-header">
      <div class="atlas-header-tools">{children}</div>
      <nav class="atlas-nav" aria-label="Atlas 导航">
        {links.map(([label, slug]) => {
          const targetSlug = slug.replace(/\/index$/, "").toLowerCase()
          const active = currentSlug === targetSlug || currentSlug.startsWith(`${targetSlug}/`)
          return (
            <a
              href={resolveRelative(fileData.slug!, slug)}
              class={active ? "active" : undefined}
              aria-current={active ? "page" : undefined}
            >
              {label}
            </a>
          )
        })}
      </nav>
    </header>
  )
}

export default (() => Header) satisfies QuartzComponentConstructor
