"use strict";

const top_swiper = new Swiper(".first-container", {
  // Optional parameters
  direction: "horizontal",
  loop: true,

  // Navigation arrows
  navigation: {
    nextEl: ".first-next",
    prevEl: ".first-prev",
  },

  slidesPerView: 5,
  spaceBetween: 20,
  speed: 600,

  breakpoints: {
    768: {
      slidesPerView: 4,
    },
    1250: {
      slidesPerView: 5,
    },
  },
});

const recipes_to_like_slider = new Swiper(".recipes-to-like-container", {
  // Optional parameters
  direction: "horizontal",
  loop: true,

  // Navigation arrows
  navigation: {
    nextEl: ".to-like-next",
    prevEl: ".to-like-prev",
  },

  slidesPerView: 5,
  spaceBetween: 20,
  speed: 600,

  breakpoints: {
    768: {
      slidesPerView: 4,
    },
    1250: {
      slidesPerView: 5,
    },
  },
});

const new_slider = new Swiper(".second-container", {
  // Optional parameters
  direction: "horizontal",
  loop: true,

  // Navigation arrows
  navigation: {
    nextEl: ".second-next",
    prevEl: ".second-prev",
  },

  slidesPerView: 5,
  spaceBetween: 20,
  speed: 600,

  breakpoints: {
    768: {
      slidesPerView: 4,
    },
    1250: {
      slidesPerView: 5,
    },
  },
});

const users_recipes_slider = new Swiper(".users-recipes-container", {
  // Optional parameters
  direction: "horizontal",
  loop: true,

  // Navigation arrows
  navigation: {
    nextEl: ".users-recipes-next",
    prevEl: ".users-recipes-prev",
  },

  slidesPerView: 5,
  spaceBetween: 20,
  speed: 600,

  breakpoints: {
    768: {
      slidesPerView: 4,
    },
    1250: {
      slidesPerView: 5,
    },
  },
});

const recbook_swiper = new Swiper(".third-container", {
  direction: "horizontal",
  loop: true,
  navigation: {
    nextEl: ".first-next",
    prevEl: ".first-prev",
  },
  slidesPerView: 5,
  spaceBetween: 20,
  speed: 600,

  breakpoints: {
    768: {
      slidesPerView: 4,
    },
    1250: {
      slidesPerView: 5,
    },
  },
});

setTimeout(function () {
  if ($("#msg").length > 0) {
    $("#msg").fadeTo(500, 0, function () {});
  }
}, 4000);

function login_user() {
  let email = document.getElementById("exampleInputUsername").value;
  let password = document.getElementById("exampleInputPassword").value;
  let csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0]
    .value;
  $("#errorlogin").html("");
  $.ajax({
    type: "POST",
    url: "/users/func_login",
    data: {
      csrfmiddlewaretoken: csrfmiddlewaretoken,
      username: email,
      password: password,
    },
    success: function (data) {
      if (data["message"] == "Success") {
        location.href = data["homeUrl"];
      } else if (data["message"] == "Denied") {
        $("#errorlogin").html("Invalid Username or Password");
      }
    },
  });
}

function combinedСallUsername() {
  validateCredentials();
  usernameErrorMessage();
}

function combinedСallPassword() {
  validateCredentials();
  passwordErrorMessage();
}

function validateCredentials() {
  var first = document.getElementById("exampleInputUsername").value.length;
  var second = document.getElementById("exampleInputPassword").value.length;

  if (first > 5 && second > 7) {
    document.getElementById("submitButton").disabled = false;
  } else {
    document.getElementById("submitButton").disabled = true;
  }
}
function usernameErrorMessage() {
  var inputLogin = document.getElementById("exampleInputUsername").value.length;

  if (inputLogin < 6) {
    $("#errorUsername").html("Minimum 6 characters").css("color", "red");
  } else {
    $("#errorUsername").html("   ");
  }
}

function passwordErrorMessage() {
  var inputPassword = document.getElementById("exampleInputPassword").value
    .length;

  if (inputPassword < 8) {
    $("#errorPassword").html("Minimum 8 characters").css("color", "red");
  } else {
    $("#errorPassword").html("   ");
  }
}

$(function () {
  $("#exampleInputUsername").on("keypress", function (e) {
    if (e.which == 32) return false;
  });
});

$(function () {
  $("#exampleInputPassword").on("keypress", function (e) {
    if (e.which == 32) return false;
  });
});
