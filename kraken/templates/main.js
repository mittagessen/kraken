const $ = (s) => document.querySelector(s)
const $$ = (s) => document.querySelectorAll(s)
const activeClass = 'active'
const hoverClass = 'hovered'

function activate(...els) { els.forEach(el => el.classList.add(activeClass)) }
function deactivate(...els) { els.forEach(el => el.classList.remove(activeClass)) }
function hoverate(...els) { els.forEach(el => el.classList.add(hoverClass)) }
function dehoverate(...els) { els.forEach(el => el.classList.remove(hoverClass)) }

document.addEventListener('DOMContentLoaded', function() {
	const uuid = $('meta[name=uuid]').getAttribute('content')
	const inputFields = $$('li[contenteditable=true]')

	const getLocalStorageId = (lineId) => `${uuid}__${lineId}`

	if (localStorage != null) {
		inputFields.forEach(function(li) {
			li.textContent = localStorage.getItem(getLocalStorageId(li.id)) || ''
		});
	}

	// focus text fields when lines/words are clicked + mouseover
	$$('a.rect').forEach(function(a) {
		const field = document.getElementById(a.getAttribute('alt'))

		a.addEventListener('click', function(e) {
			e.preventDefault();
			activate(a, field)
			field.focus()
		})

		a.addEventListener('mouseover', () => hoverate(a, field))
		a.addEventListener('mouseout', () => dehoverate(a, field))
	})

	// create mouseover effect on text fields
	inputFields.forEach(function(field) {
		const a = $(`a.rect[alt="${field.id}"]`)

		field.addEventListener('mouseover', () => hoverate(a, field))
		field.addEventListener('mouseout', () => dehoverate(a, field))
		field.addEventListener('focus', () => activate(a, field))
		field.addEventListener('blur', () => deactivate(a, field))

		field.addEventListener('keydown', function(e) {
			if (e.which != 13) return
			e.preventDefault()

			field.classList.add('corrected')
			field.nextElementSibling.focus()
		})

		field.addEventListener('keyup', function () {
			localStorage.setItem(getLocalStorageId(field.id), field.textContent)
		})
	})


	// serializing the DOM to a file
	const button = $('button.download > a')
	button.addEventListener('click', function(e) {
		const path = window.location.pathname
		button.setAttribute('href', 'data:text/html,' + encodeURIComponent(document.documentElement.outerHTML))
		button.setAttribute('download', path.substr(path.lastIndexOf('/') + 1))
	})
})
