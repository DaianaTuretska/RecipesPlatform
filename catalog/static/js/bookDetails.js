const timeToLiveMin = 10;
const starStorageKey = "bookDetailsRatingStar";
let ratingStars = document.querySelectorAll(".rating__control");

$(document).ready(function () {
  let endpoint = document.getElementById("recipe-reviews").getAttribute("url");
  let csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0]
    .value;

  $.ajax({
    method: "GET",
    url: endpoint,
    headers: {
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    dataType: "json",
    success: fillCommentsList,
  });
});

ratingStars.forEach((ratingStar) => {
  ratingStar.addEventListener("click", function () {
    setWithExpiry(starStorageKey, parseInt(ratingStar.value), timeToLiveMin);
  });
});

function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
          const cookie = cookies[i].trim();
          // Does this cookie string begin with the name we want?
          if (cookie.substring(0, name.length + 1) === (name + '=')) {
              cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
              break;
          }
      }
  }
  return cookieValue;
}

function setWithExpiry(key, value, ttl) {
  let now = new Date();
  let item = {
    value: value,
    expiry: now.getTime() + ttl * 60000,
  };
  localStorage.setItem(key, JSON.stringify(item));
}

function getWithExpiry(key) {
  const itemStr = localStorage.getItem(key);
  if (!itemStr) {
    return null;
  }
  let item = JSON.parse(itemStr);
  let now = new Date();
  if (now.getTime() > item.expiry) {
    localStorage.removeItem(key);
    return null;
  }
  return item.value;
}

function saveStarRating(url) {
  let starValue = getWithExpiry(starStorageKey);
  let csrfmiddlewaretoken = getCookie("csrftoken");
  let data = {
    rating: starValue
  };

  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfmiddlewaretoken
    },
    body: JSON.stringify(data)
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    // You can handle success here if needed
    location.reload();
  })
  .catch(error => {
    console.error('There was a problem with the fetch operation:', error);
  });
}



function addMouseEventsForComments() {
  let commentArea = document.querySelectorAll("#commentArea");
  // console.log(commentArea)
  commentArea.forEach((comment) => {
    comment.addEventListener("mouseover", function (event) {
      comment.querySelector("#commentButton").style.display = "block";
    });
    comment.addEventListener("mouseout", function (event) {
      comment.querySelector("#commentButton").style.display = "none";
    });
  });
}

function fillCommentsList(list) {
  let userId = document.getElementById("userId").getAttribute("value");
  let userRole = document.getElementById("userRole").getAttribute("value");
  let errorMessage = list.error;
  if(errorMessage){
      showError(errorMessage);
  }
  const data = JSON.parse(list.reviews);
  $("#commentList").empty();

  for (let i = 0; i < data.length; i++) {
    const buttonClass = `comment-delete-button-${i}`;
    const buttonClassAdmin = `comment-restore-button-${i}`;
    const userDeleteButton = `<a id="commentButton" style="float: right; width: 40px; display: none;"
                   class="btn btn-outline-danger ${buttonClass}">×</a>`;

    const adminRestoreButton = `<a id="commentButton" style="float: right; width: 100px; display: none;"
                   class="btn btn-outline-info ${buttonClassAdmin}">Restore</a>`;

    const date = new Date(data[i].created_at).toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const regularUserComment =
      data[i].status === "active" && `
          <div class="card mt-2 pl-2 pr-2 pt-2" id="commentArea" >
          <div style="font-size: 15pt; background: white;">
                    ${data[i].user_firstname} ${data[i].user_lastname}
                    ${userId == data[i].user_id && userDeleteButton ? userDeleteButton : ""}
             </div>
          <div class="card-body">
                        <blockquote class="blockquote mb-0">
                            <p style="font-size: 14pt">${data[i].comment}</p>
                            <footer style="font-size: 10pt" class="blockquote-footer"><cite
                            title="Source Title">Published ${date}</cite></footer>
                </blockquote>
                     </div>
            </div>`;

    const adminComment = `
          <div class="card mt-2 pl-2 pr-2 pt-2" id="commentArea" >
          <div style="font-size: 15pt; background: white;">
                    ${data[i].user_firstname} ${data[i].user_lastname}
                    ${
                      data[i].status === "active"
                        ? userDeleteButton
                        : adminRestoreButton
                    }


             </div>
          <div class="card-body">
                        <blockquote class="blockquote mb-0">
                            <p style="font-size: 14pt">${data[i].comment}</p>
                            <footer style="font-size: 10pt" class="blockquote-footer"><cite
                            title="Source Title">Published ${date}</cite></footer>
                </blockquote>
                     </div>
            </div>`;

    const content = userRole === "admin" ||  userRole === "moderator" ? adminComment : regularUserComment;

    if (content) {
      $("#commentList").append(content);
      if (userRole == "admin" ||  userRole == "moderator") {
      change_comment_status(`.${buttonClass}`,
      `/change_review_status/${data[i].id}/`,
      "inactive",
      `/reviews/${data[i].recipe_id}/`)
      change_comment_status(`.${buttonClassAdmin}`,
      `/change_review_status/${data[i].id}/`,
      "active",
      `/reviews/${data[i].recipe_id}/`)
      } else {
      change_comment_status(`.${buttonClass}`,
      `/change_review_status/${data[i].id}/`,
      "inactive",
      `/reviews/${data[i].recipe_id}/`)
      }
    }
    addMouseEventsForComments();
  }
}

addMouseEventsForComments();

function add_comment() {
  let textComment = CKEDITOR.instances.exampleFormControlTextarea1.getData();

  let endpoint = document
    .getElementById("recipe-reviews")
    .getAttribute("url");
  let csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0]
    .value;

  $.ajax({
    method: "POST",
    url: endpoint,
    headers: {
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    data: {
      "text-comment": textComment,
    },
    dataType: "json",
    success: fillCommentsList
  });
  console.log("add com post")
}

function change_comment_status(classname, url, new_status, getListURL) {
    let elementButton = document.querySelector(classname)
    if (elementButton){
    let csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0]
    .value;
    elementButton.onclick = function (){
                 $.ajax({
    method: "POST",
    url: url,
    headers: {
      "X-CSRFToken": csrfmiddlewaretoken,
    },
    data: {
      status: new_status
    },
    dataType: "json",
    complete:function(res){

    if (res.status === 200) {
        $.ajax({
        method:"GET",
        url: getListURL,
        headers: {
          "X-CSRFToken": csrfmiddlewaretoken,
        },
    dataType: "json",
    success: fillCommentsList
        });
        }
    }
  });
  }
}
}

function showError(errorMessage){
    let errorDiv = document.getElementById('errorMessage');
    errorDiv.style.opacity = '1';
    errorDiv.classList.add('alert-danger');
    errorDiv.innerHTML = errorMessage;
    setTimeout(function (){
      $(errorDiv).fadeTo(500, 0.0, 'linear');
    }, 5000);
}

function submitForm(event, url) {
  event.preventDefault(); // Prevents the default form submission behavior
  saveStarRating(url); // Call your function to handle the form submission
}