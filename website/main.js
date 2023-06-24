import { createBenchmarksV } from './benchmarks/main.js'

async function main() {
    window.addEventListener('hashchange', () => {
        location.reload()
    })

    const url = location.hostname === 'tju01.github.io'
        ? 'https://raw.githubusercontent.com/tju01/ilm-eval/main/reports'
        : './reports'

    document.body.textContent = 'Loading. May take a few seconds...'
    const benchmarksV = await createBenchmarksV(url)
    document.body.replaceChildren(benchmarksV)
}

main()
